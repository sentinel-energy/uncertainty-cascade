from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple, Union

from glom import glom, Iter
import pandas as pd
import xarray as xr

from errapids.io import _path_t, decode_fname
from errapids.ons import pv_batt_lvls


class Metrics:
    """Parses the Calliope model result and provides summary metrics

    A Calliope model result encodes the factor/categorical variables for some
    of the metrics by concatenating them: location, technology, [carrier].
    Others combination are left as is: e.g. carrier, cost, technology.  This
    class parses the concatenated factor/categorical variables, as well as
    resolves their order into one definite order: cost, location, technology,
    carrier.

    Cost is always "summed" over as usually there is only one type of cost
    (monetary).  There can be other kinds of costs, like emissions, in which
    case this class has to be updated.

    If a factor/level is absent, it is simply skipped in the summary.

    >>> metrics = Metrics.from_netcdf("path/to/model_result.nc")
    >>> metrics["energy_cap"]
    # a dataframe

    """

    @classmethod
    def from_netcdf(cls, fpath: _path_t) -> "Metrics":
        """Read the saved model result from a NetCDF file"""
        return cls(xr.open_dataset(fpath))

    def __init__(self, dst: xr.Dataset):
        self._dst = dst.filter_by_attrs(is_result=1)

    def __repr__(self) -> str:
        return repr(self._dst)

    def __getitem__(self, metric: str) -> pd.Series:
        """Access a metric from the model result"""
        darr = self._dst[metric]
        if "costs" in darr.dims:  # costs: redundant, usually only monetary
            darr = darr.sel(costs="monetary")
        arr = darr.to_pandas()  # series or dataframe (when timeseries)

        if "carriers" in darr.dims and len(darr.dims) > 1:  # when not concatenated
            arr = arr.stack()  # dataframe -> series

        if arr.ndim == 2:  # timesteps: aggregated
            arr = arr.mean(axis=1)  # dataframe -> series
        assert arr.ndim == 1

        # factors of interest: location, technology, [carrier]
        arr.index = self.__idx__(arr.index)
        arr.name = metric
        return arr

    def __idx__(self, _idx: pd.Index) -> Union[pd.Index, pd.MultiIndex]:
        if isinstance(_idx, pd.MultiIndex):  # carrier, technology
            if _idx.names.index("carriers") < _idx.names.index("techs"):
                _idx = _idx.swaplevel()
            _idx.names = ["technology", "carrier"]
            return _idx
        elif _idx.name == "carriers":  # carrier
            _idx.name = "carrier"
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

    @classmethod
    def from_dir(cls, dpath: _path_t, glob: str) -> "ScenarioGroups":
        return cls.from_netcdfs(Path(dpath).glob(glob))

    @classmethod
    def from_netcdfs(cls, fpaths: Iterable[_path_t]) -> "ScenarioGroups":
        return cls(map(xr.open_dataset, fpaths))

    @classmethod
    def __prettify__(cls, name: str) -> str:
        factor = pv_batt_lvls[int(name[-1]) - 1]
        return name[:-1] + "{:03d}".format(int(factor * 100))

    @classmethod
    def __unpack_overrides__(cls, overrides: str) -> Tuple[str, ...]:
        lvls = overrides.split(";")[:-1]
        # override order: 1) building heating, EV charging, 2) PV, 3) battery
        heating, charging = decode_fname(lvls[0])
        pv, battery = lvls[1:]
        return heating, charging, cls.__prettify__(pv), cls.__prettify__(battery)

    def __init__(self, scenarios: Iterable[xr.Dataset]):
        self._scenarios = {
            dst.scenario: (
                self.__unpack_overrides__(dst.applied_overrides),
                Metrics(dst),
            )
            for dst in scenarios
        }
        self.varnames = glom(
            self._scenarios.values(), (Iter("1._dst.data_vars").first(), list)
        )

    def __repr__(self) -> str:
        return "\n---\n".join(
            f"Scenario: {v[0]}\n{v[1]}" for v in self._scenarios.values()
        )

    def __getitem__(self, metric: str) -> pd.Series:
        df = pd.concat(
            [
                metrics[metric]
                .to_frame()
                .assign(**{col: lvl for col, lvl in zip(self.idxcols, lvls)})
                for lvls, metrics in self._scenarios.values()
            ],
            axis=0,
        ).set_index(self.idxcols, append=True)
        nlvls = df.index.nlevels
        # 4 levels of scenarios: heating, EV, PV, battery; move them forward
        new_order = list(chain(range(nlvls)[-4:], range(nlvls)[:-4]))
        return df.reorder_levels(new_order)[metric]


def fraction_by_level(arr: pd.Series, lvl: Union[int, str]) -> pd.Series:
    """Calculate the fraction out of the total for a given level"""
    _lvl = arr.index.names.index(lvl) if isinstance(lvl, str) else lvl
    # sum over any deeper levels
    numerator = arr.groupby(level=list(range(_lvl + 1))).sum()
    # sum over desired level to get reference
    denominator = arr.groupby(level=list(range(_lvl))).sum()
    numerator, denominator = numerator.align(denominator, method="ffill")
    return numerator / denominator
