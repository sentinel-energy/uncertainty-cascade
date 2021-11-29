from itertools import chain
from pathlib import Path
from textwrap import indent
from typing import Iterable, Dict, Literal, List, Tuple, Union, overload

from glom import glom, Iter
import numpy as np
import pandas as pd
import xarray as xr

from friendly_data.helpers import noop_map

from errapids.err import baselines
from errapids.err import _isgrouped
from errapids.err import notrans
from errapids.err import scenario_deltas
from errapids.err import DFSeries_t
from errapids.err import sum_all
from errapids.io import _path_t
from errapids.ons import pv_batt_lvls
from errapids.tseries import daily_summary

_flevel_aliases = {
    "high": ["high", "hi"],
    "mid": ["mid", "medium"],
    "low": ["low", "lo"],
}

ix = pd.IndexSlice


def qual_name(name: str, trans: bool = False) -> str:
    name = (
        name.replace("carrier", "electricity")
        .replace("_con", "_consumption")
        .replace("_prod", "_production")
        .replace("var", "variable")
        .replace("_cap", "_capacity")
        .replace("_", " ")
    )
    if "cost" in name:
        name = f"{name} (bn EUR)"
    elif "capacity" in name or "storage" in name:
        name = f"{name} (100 GW)"
    elif any(map(lambda i: i in name, ("consumption", "production", "electricity"))):
        name = f"{name} (100 GWh)"
    elif "area" in name:
        name = f"{name} (10k km²)"

    if trans:
        name = name.replace("consumption", "export").replace("production", "import")
    return name


def decode_fname(fname: str) -> Tuple[str, str]:
    tokens = fname.lower().split("_")

    def remap(token: str) -> str:
        key = [k for k, v in _flevel_aliases.items() if token in v]
        assert key
        return key[0]

    # building heating, charging profile
    heating = remap(tokens[0])
    charging = remap(tokens[2])
    # return f"{heating}_{charging}"
    return heating, charging


def decode_link(_idx: pd.Index):
    transport = _idx.get_level_values("technology").str.startswith("ac_transmission")
    idx = _idx[transport].str.split(":", expand=True)
    idx.names = [_idx.name, "link"]
    return idx, transport


def prettify_costs(name: str, pretty: bool) -> Union[str, int]:
    factor = pv_batt_lvls[int(name[-1]) - 1]
    if pretty:
        return name[:-1] + "{:03d}".format(int(factor * 100))
    else:
        return int(factor * 100)


def ensure_frame(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame()
    else:
        return data


class Metrics:
    """Parses the Calliope model result and provides summary metrics

    A Calliope model result encodes the factor/categorical variables for some
    of the metrics by concatenating them: location, technology, [carrier].
    Other combinations are left as is: e.g. carrier, cost, technology.  This
    class parses the concatenated factor/categorical variables, as well as
    resolves their order into one definite order: cost, location, technology,
    carrier.

    Cost is always "summed" over as usually there is only one type of cost
    (monetary).  There can be other kinds of costs, like emissions, in which
    case this class has to be updated.

    If a factor/level is absent, it is simply skipped in the summary.

    >>> metrics = Metrics.from_netcdf("path/to/model_result.nc")
    >>> metrics["energy_cap"]
    # a pandas.Series

    """

    @classmethod
    def from_netcdf(cls, fpath: _path_t) -> "Metrics":
        """Read the saved model result from a NetCDF file"""
        return cls(xr.open_dataset(fpath))

    def is_tseries(self, metric: str) -> bool:
        return "timesteps" in getattr(self._dst, metric).coords

    def __init__(self, dst: xr.Dataset):
        self._dst = dst.filter_by_attrs(is_result=1)
        self.time_varying = [v for v in self._dst.data_vars if self.is_tseries(v)]

    def __repr__(self) -> str:
        return repr(self._dst)

    def __getitem__(self, metric: str) -> pd.Series:
        return self.get(metric, summarise=True)

    @overload
    def get(self, metric: str, summarise: Literal[True]) -> pd.Series:
        ...

    @overload
    def get(self, metric: str, summarise: Literal[False]) -> pd.DataFrame:
        ...

    def get(self, metric: str, summarise: bool) -> Union[pd.Series, pd.DataFrame]:
        """Access a metric from the model result"""
        darr = self._dst[metric]
        if "costs" in darr.dims:  # costs: redundant, usually only monetary
            darr = darr.sel(costs="monetary")
        arr = darr.to_pandas()  # series or dataframe (when timeseries)
        # when timeseries, "wide" format, i.e. timesteps as columns

        if "carriers" in darr.dims and len(darr.dims) > 1:  # when not concatenated
            arr = arr.stack()  # dataframe -> series

        if summarise:
            if arr.ndim == 2:  # timesteps: aggregated
                arr = arr.mean(axis=1)  # dataframe -> series
            assert arr.ndim == 1

        # factors of interest: location, technology, [carrier]
        arr.index = self.__idx__(arr.index)
        arr.name = metric
        return arr

    def __idx__(self, _idx: pd.Index) -> Union[pd.Index, pd.MultiIndex]:
        if isinstance(_idx, pd.MultiIndex):  # carrier, technology
            try:
                if _idx.names.index("carriers") < _idx.names.index("techs"):
                    _idx = _idx.swaplevel()
            except ValueError:
                # FIXME: some kind of caching is happening on repeat calls
                if _idx.names.index("carrier") < _idx.names.index("techs"):
                    _idx = _idx.swaplevel()
            _idx.names = ["technology", "carrier"]
            return _idx
        elif _idx.name == "carriers":  # carrier
            _idx.name = "carrier"
            return _idx
        elif _idx.name == "carrier":
            # FIXME: some kind of caching is happening on repeat calls
            return _idx

        # parse concatenated location, technology, [carrier] values
        msg = f"{_idx.name}: unsupported index"
        # index name: loc_techs_*, loc_tech_carriers_*
        if "loc_tech" not in _idx.name:
            raise ValueError(msg)
        idx = _idx.str.split("::", expand=True)
        if "carrier" in _idx.name:
            if idx.nlevels == 3:
                idx.names = ["region", "technology", "carrier"]
            else:
                raise ValueError(msg)
        else:
            if idx.nlevels == 2:
                idx.names = ["region", "technology"]
            else:
                raise ValueError(msg)
        return idx


class ScenarioGroups:
    """Hierachically group model results for different scenarios"""

    idxcols = "heating,EV,PV,battery".split(",")
    _banned = ["capacity_factor"]
    derived = ["carrier_prod_share", "capacity_factor"]

    @classmethod
    def from_dir(cls, dpath: _path_t, glob: str, **kwargs) -> "ScenarioGroups":
        return cls.from_netcdfs(Path(dpath).glob(glob), **kwargs)

    @classmethod
    def from_netcdfs(cls, fpaths: Iterable[_path_t], **kwargs) -> "ScenarioGroups":
        return cls(map(xr.open_dataset, fpaths), **kwargs)

    @classmethod
    def __unpack_overrides__(cls, overrides: str, pretty: bool) -> Tuple:
        lvls = overrides.split(";")[:-1]
        # override order: 1) building heating, EV charging, 2) PV, 3) battery
        heating, charging = decode_fname(lvls[0])
        pv, battery = lvls[1:]
        return (
            heating,
            charging,
            prettify_costs(pv, pretty),
            prettify_costs(battery, pretty),
        )

    # @property
    # def alias(self):
    #     return self._alias

    # @alias.setter
    # def alias(self, alias: Dict[str, str]):
    #     self._alias = noop_map(alias if alias else {"energy_cap": "nameplate_capacity"})
    #     self._rev_alias = noop_map((v, k) for k, v in self._alias.items())

    # @property
    # def ralias(self):
    #     return self._rev_alias

    def __init__(
        self, scenarios: Iterable[xr.Dataset], pretty: bool = True, alias: Dict = {}
    ):
        # self.alias = alias
        self._scenarios = {
            dst.scenario: (
                self.__unpack_overrides__(dst.applied_overrides, pretty),
                Metrics(dst),
            )
            for dst in scenarios
        }
        self.varnames = glom(
            self._scenarios.values(),
            (
                Iter("1._dst.data_vars").first(),
                list,
                Iter().filter(lambda v: v not in self._banned).all(),
            ),
        )
        dst = glom(self._scenarios.values(), Iter("1").first())
        self.varnames_ts = [var for var in self.varnames if dst.is_tseries(var)]
        self.metrics = self.varnames + self.derived

    def __repr__(self) -> str:
        return "\n---\n".join(
            f"Scenario: {v[0]}\n{v[1]}" for v in self._scenarios.values()
        )

    def __getitem__(self, metric: str) -> pd.Series:
        if metric in self.varnames:
            return self.get(metric, summarise=True)[metric]
        elif metric in self.derived:
            return self.derive(metric, summarise=True)[metric]
        else:
            raise KeyError(f"{metric}: unknown metric")

    def get(self, metric: str, summarise: bool) -> pd.DataFrame:
        if metric in self._banned:
            raise ValueError(f"{metric}: derived metric, use `derive(..)`")
        df = pd.concat(
            [
                ensure_frame(metrics.get(metric, summarise)).assign(
                    **{col: lvl for col, lvl in zip(self.idxcols, lvls)}
                )
                for lvls, metrics in self._scenarios.values()
            ],
            axis=0,
        ).set_index(self.idxcols, append=True)
        nlvls = df.index.nlevels
        # 4 levels of scenarios: heating, EV, PV, battery; move them forward
        new_order = list(chain(range(nlvls)[-4:], range(nlvls)[:-4]))
        return df.reorder_levels(new_order)

    def derive(self, metric: str, summarise: bool = True) -> pd.DataFrame:
        if metric == "carrier_prod_share":
            return technology_share(self["carrier_prod"]).rename(metric).to_frame()
        elif metric == "capacity_factor":
            prod = self.get("carrier_prod", summarise)  # maybe a time-series
            # capacity = self["nameplate_capacity"]
            capacity = self["energy_cap"]
            res = _ratio(prod, capacity, scale=3)
            if res.shape[1] == 1:
                return res.rename(columns={"carrier_prod": metric})
            else:
                return res
        else:
            raise RuntimeError("shouldn't be here, panic!")


def technology_share(arr: pd.Series) -> pd.Series:
    """Calculate the technology share"""
    arr = notrans(arr)
    _lvl = arr.index.names.index("technology")
    # sum over any deeper levels
    numerator = arr.groupby(level=list(range(_lvl + 1))).sum()
    # sum over desired level to get reference
    denominator = arr.groupby(level=list(range(_lvl))).sum()
    numerator, denominator = numerator.align(denominator, method="ffill")
    return numerator / denominator


def _ratio(num: DFSeries_t, den: pd.Series, *, scale: int) -> DFSeries_t:
    num, den = num.align(den, join="inner", axis=0)
    if num.ndim > 1:
        return num.div(scale).div(den, axis=0).dropna()
    else:
        return num.div(scale).div(den).dropna()


def ratio_sum(
    num: DFSeries_t,
    den: pd.Series,
    lvl: str,
    *,
    scale: int,
) -> DFSeries_t:
    """Calculate the ratio of two metrics summing over a given level"""
    # not using set to preserve order
    lvls = [i for i in num.index.names if i in den.index.names]
    if lvl not in lvls:
        raise ValueError(f"{lvl=}: not in index, cannot sum")
    lvls.pop(lvls.index(lvl))

    # sum over any deeper levels
    num = num.groupby(level=lvls).sum()
    den = den.groupby(level=lvls).sum()
    return _ratio(num, den, scale=scale)


def pan_eu_cf(prod, cap, sumover: str) -> pd.DataFrame:
    if sumover not in ("region", "technology"):
        raise ValueError(f"{sumover}: unknown level")
    other = "region" if sumover == "technology" else "technology"
    prod = notrans(prod)
    cap = notrans(cap)
    cf = ratio_sum(prod, cap, sumover, scale=3).rename(f"capacity_factor_{other}")
    return scenario_deltas(cf)


def pan_eu_prod_share(arr: pd.Series) -> pd.DataFrame:
    # swap region, technology
    lvls = list(arr.index.names)
    ridx = lvls.index("region")
    tidx = lvls.index("technology")
    lvls[ridx], lvls[tidx] = lvls[tidx], lvls[ridx]
    arr = technology_share(arr.reorder_levels(lvls))
    return scenario_deltas(arr.rename("carrier_prod_share_all"))


def pan_eu_sum(
    df: pd.Series, region_grp: bool = False, technology_grp: str = "prod"
) -> pd.DataFrame:
    df = sum_all(df, technology_grp, region_grp)
    return scenario_deltas(df)


def pan_eu_sum_ts(
    df: pd.DataFrame, region_grp: bool = False, technology_grp: str = "prod"
):
    df = sum_all(df, technology_grp, region_grp)
    return daily_summary(df)


def agg_scenario(
    data,
    scenario: str,
    technology_grp: str = "prod",
    region_grp: Union[bool, str] = False,
    metrics: List[str] = [],
) -> pd.DataFrame:
    def _get(arr):
        if "carrier" in arr.index.names:
            arr = arr.droplevel("carrier")
        return sum_all(arr, technology_grp, region_grp)

    if len(metrics) == 0:
        metrics = ["carrier_con", "carrier_prod", "cost", "energy_cap"]

    base = {k: v for k, v in baselines.items() if not _isgrouped(k)}
    if _isgrouped(scenario):
        unpinned = set(base) - set(baselines[scenario])
        base.update((sc, slice(None)) for sc in unpinned)
    else:
        base[scenario] = slice(None)

    idx = ix[tuple(base.values())]
    df = pd.concat([_get(data[m]).loc[idx] for m in metrics], axis=1)

    df.index = df.index.droplevel(
        [n for n in df.index.names if len(df.index.get_level_values(n).unique()) == 1]
    )
    return df.T
