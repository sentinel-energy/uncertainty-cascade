from itertools import cycle, islice
from logging import getLogger
from typing import Dict, List, Sequence, Tuple, TypeVar, Union
from holoviews.core.spaces import HoloMap

import numpy as np
import pandas as pd
from tqdm import tqdm

import holoviews as hv
from holoviews import opts

from errapids.metrics import metric_as_dfs

logger = getLogger(__name__)


# region groups
rgroups = {
    "nordic": ["DNK", "FIN", "NOR", "SWE"],  # scandinavia
    "baltic": ["EST", "LTU", "LVA"],  # east of scandinavia
    "poland": ["POL"],  # poland
    "west": ["DEU", "FRA", "NLD", "LUX"],  # west
    "med": ["ESP", "ITA", "GRC", "PRT"],  # mediterranean
    "isles": ["GBR", "IRL"],  # british isles
    "balkan": ["MKD", "ROU", "SRB", "SVN", "HRV"],  # balkan with coast line
    "landlocked": ["SVK", "HUN", "CZE"],  # land locked
}
# scenario baselines
baselines = {
    "heating": "mid",
    "EV": "low",
    "PV": 100,
    "battery": 100,
    "demand": ["PV", "battery"],  # order same as in index
    "cost": ["heating", "EV"],
    "all": [],  # no pins
}


def _isgrouped(scenario: str) -> bool:
    """Check if scenario is a meta (grouped) scenario

    Convention: for a regular scenario, the value is the baseline value.  For a
    grouped scenario, it is a list of other scenarios that are pinned to their
    baselines.

    """
    return isinstance(baselines[scenario], list)


def qsum(data: Sequence) -> float:
    """Add in quadrature, typically uncertainties

    .. math::

    \\sqrt{\\sum_{i=1}^{n} x^2_i}

    """
    return np.sqrt(np.square(data).sum())


DFSeries_t = TypeVar("DFSeries_t", pd.DataFrame, pd.Series)


def lvl_filter(
    df: DFSeries_t, lvl: str, token: str, invert: bool = True, reverse: bool = False
) -> DFSeries_t:
    """Filter a dataframe by the values in the index at the specified level

    Parameters
    ----------
    df : Union[pandas.DataFrame, pandas.Series]
        Dataframe/series

    lvl : str
        Level name

    invert : bool (default: True)
        Whether to filter out (default) matches, or select matches

    reverse : bool (default: False)
        Whether to match the start (default) or end of the value

    Returns
    -------
    Union[pandas.DataFrame, pandas.Series]
        The filtered dataframe/series

    """
    if reverse:
        sel = df.index.get_level_values(lvl).str.endswith(token)
    else:
        sel = df.index.get_level_values(lvl).str.startswith(token)
    return df[~sel if invert else sel]


def marginalise(df: pd.DataFrame, scenario: str) -> Union[pd.DataFrame, pd.Series]:
    """Marginalise all scenarios except the one specified.

    Pin all scenarios except the one specified to their baselines, and
    calculate the uncertainty based on the remaining scenarios.  The lower and
    upper values are put in columns `errlo` and `errhi`.  The returned
    dataframe has the columns: `<original column>`, `errlo`, and `errhi`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    scenario : str
        The scenario name; should be one of the scenarios defined in `baselines`

    Returns
    -------
    NDFrame
        The marginalised dataframe/series

    """
    colname = df.columns[0]
    if scenario not in baselines:
        raise ValueError(f"unknown {scenario=}: must be one of {list(baselines)}")

    # pin other scenarios
    pins = (
        [sc for sc in baselines[scenario]]
        if _isgrouped(scenario)
        else [sc for sc in baselines if sc != scenario and not _isgrouped(sc)]
    )
    if pins:
        query = [f"{sc} == {baselines[sc]!r}" for sc in pins]
        df = df.query(" & ".join(query))
        df.index = df.index.droplevel(pins)

    unpinned = [
        baselines[sc] for sc in baselines if sc not in pins and not _isgrouped(sc)
    ]
    df = df.unstack(list(range(len(unpinned))))
    baseline = df[(colname, *unpinned) if unpinned else colname]
    names = [colname, "errlo", "errhi"]
    if df.ndim > 1:
        deltas = [(df.apply(f, axis=1) - baseline).abs() for f in (np.min, np.max)]
        df = pd.concat([baseline, *deltas], axis=1)
        df.columns = names
        return df
    else:  # series
        deltas = [np.abs(f(df) - baseline) for f in (np.min, np.max)]
        return pd.Series([baseline, *deltas], index=names)


def scenario_deltas(df: pd.DataFrame, istransmission: bool = False) -> pd.DataFrame:
    """Calculate uncertainties for different scenarios.

    This adds a new level in the index for every scenario.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    istransmission : bool (default: False)
        Whether to consider transmission data or not

    Returns
    -------
    pandas.DataFrame

    """
    if "technology" in df.index.names:
        df = lvl_filter(df, "technology", "ac_transmission", invert=not istransmission)
    if df.empty:
        raise ValueError(f"{istransmission=}: returned empty dataframe")
    if "carrier" in df.index.names:  # redundant for us
        df.index = df.index.droplevel("carrier")

    # FIXME: instead of iterating, could do it by generating the right indices
    # in marginalise
    dfs = []
    for scenario in baselines:
        _df = marginalise(df, scenario)
        if isinstance(_df, pd.DataFrame):
            null = _df.apply(lambda row: np.isclose(row, 0).all(), axis=1)
            _df = (
                _df[~null].assign(scenario=scenario).set_index("scenario", append=True)
            )
        else:  # series
            _df = pd.DataFrame([_df], index=pd.Index([scenario], name="scenario"))
        dfs.append(_df)
    df = pd.concat(dfs, axis=0)
    if isinstance(df.index, pd.MultiIndex):
        return df.reindex(index=list(baselines), level="scenario")
    else:
        return df.reindex(index=list(baselines))


def facets(df: pd.DataFrame, region: str, stack: str) -> Tuple:
    """Create a bar chart, and a faceted set of uncertainties

    The number of countries are limited to the ``region`` defined in `rgroups`.
    The index level (or dimension) ``stack`` is "stacked" in the summary plot,
    and summed over in the undertainty plots.  The unceratainty plots are
    faceted over the "remaining" index level (or dimension) - "region" or
    "technology".

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    region : str
        Region group, as defined in `rgroups`

    stack : str
        Index level (or dimension) to stack/sum over

    Returns
    -------
    Tuple[holoviews.Bars, holoviews.HoloMap, holoviews.HoloMap]
        The plot elements, the holomaps contain faceted uncertainty spreads,
        and the corresponding tables

    """
    name = df.columns[0]

    if not isinstance(df.index, pd.MultiIndex):
        assert df.index.name == "scenario"
        rmax = 1.2 * df[[name, "errhi"]].sum(axis=1).max()
        bars = hv.Bars(df, vdims=[name], kdims=["scenario"]).opts(
            tools=["hover"], ylim=(0, rmax)
        )
        errs = hv.Spread(df, vdims=[name, "errlo", "errhi"], kdims=["scenario"]).opts(
            fill_alpha=0.7
        ) * hv.HLine(df.iloc[0, 0])
        return bars, errs, hv.Table(df.reset_index())

    if stack and stack not in df.index.names:
        raise ValueError(f"{name}: {stack=} not present")

    if region and "region" in df.index.names:
        grp = rgroups[region]  # noqa, choose region, implicitly used in df.query(..)
        df = df.query("region in @grp")
    elif region:
        logger.warning(f"{name}: no region level in index, {region=} ignored")
    else:
        logger.info(f"{name}: no region level in index")

    grpd = df.groupby(df.index.names.difference([stack])).agg(
        {name: np.sum, "errlo": qsum, "errhi": qsum}
    )
    if isinstance(grpd.index, pd.MultiIndex):
        grpd = grpd.reindex(index=list(baselines), level="scenario")
    else:
        grpd = grpd.reindex(index=list(baselines))

    space = 1.7 if stack == "technology" else 1.2
    rmax = space * grpd[[name, "errhi"]].sum(axis=1).max()

    # maintain original order
    kdims = df.index.names.difference(["scenario"])
    if stack:
        stackidx = kdims.index(stack)
        kdims = list(islice(cycle(kdims), stackidx + 1, stackidx + 1 + len(kdims)))
    bars = hv.Bars(df.query("scenario == 'all'"), vdims=[name], kdims=kdims).opts(
        stacked=bool(stack), tools=["hover"], ylim=(0, rmax)
    )

    faceted_lvl = grpd.index.names[0]
    if isinstance(grpd.index, pd.MultiIndex):
        lvl_vals = grpd.index.levels[grpd.index.names.index(faceted_lvl)]
    else:
        lvl_vals = grpd.index
    errs = HoloMap(
        {
            val: (
                hv.Spread(
                    grpd.loc[val, :], vdims=list(grpd.columns), kdims=["scenario"]
                ).opts(fill_alpha=0.7, ylim=(0, rmax))
                * hv.HLine(grpd.loc[val, name].iloc[0])
            )
            for val in lvl_vals
        },
        kdims=[faceted_lvl],
    )
    tbl = HoloMap(
        {
            val: hv.Table(df.query(f"{faceted_lvl} == @val").reset_index())
            for val in lvl_vals
        },
        kdims=[faceted_lvl],
    )
    return bars, errs, tbl


def layout(bars, errs, tbl, height):
    layout = (
        hv.Layout(bars + errs + tbl)
        .opts(toolbar="right", height=height)
        .opts(
            opts.Bars(width=2 * height),
            opts.Spread(width=height),
            opts.Table(width=2 * height),
        )
        .cols(2)
    )
    return layout


class plotmanager:
    """Manage and render the plots

    Reads and processes the data on instantiation.  The plots are generated by
    calling :meth:`plot`.  All plots can be written to a directory with
    :meth:`write`.

    """

    def __init__(self, datadir: str, glob: str, istransmission: bool = False):
        self._dfs = {
            df.columns[0]: scenario_deltas(df, istransmission)
            for df in metric_as_dfs(datadir, glob, pretty=False)
        }

    @property
    def metrics(self):
        return list(self._dfs)

    @property
    def regions(self) -> List[str]:
        return list(rgroups)

    def plot(
        self, metric: str, region: str, stack: str, height: int = 400
    ) -> hv.Layout:
        """Render plot

        Parameters
        ----------
        metric : str
            Metric to plot

        region : str
            Region subset to include in plot

        stack : str
            Index level to stack

        height : int (default: 400)
            Height of individual plots, everything else is scaled accordingly

        Returns
        -------
        hvplot.Layout
            A 2-column ``hvplot.Layout`` with the different facets of the plot

        """
        bars, errs, tbl = facets(self._dfs[metric], region, stack)
        return layout(bars, errs, tbl, height)

    def write(self, plotdir: str):
        """Write all plots to a directory

        Parameters
        ----------
        plotdir : str
            Plot directory

        """
        pbar = tqdm(self.metrics)
        for metric in pbar:
            pbar.set_description(f"{metric}")
            if "total" in metric or "systemwide" in metric:
                hv.save(self.plot(metric, "", ""), f"{plotdir}/{metric}.html")
            else:
                pbar2 = tqdm(rgroups)
                for region in pbar2:
                    pbar2.set_description(f"{region=}")
                    for stack in ("region", "technology"):
                        plots = self.plot(metric, region, stack)
                        hv.save(plots, f"{plotdir}/{metric}_{region}_{stack}.html")
