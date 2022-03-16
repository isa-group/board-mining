from tkinter.tix import InputOnly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import pm4py
import networkx as nx

def log_info(df):
    log_info = {}
    log_info["events"] = len(df)
    log_info["attribs"] = len(df.columns)
    log_info["cards"] = df["data.card.id"].nunique() if "data.card.id" in df.columns else 0
    log_info["lists"] = df["data.list.id"].nunique() if "data.list.id" in df.columns else 0
    log_info["list_first_create"] = df[df["type"]=="createList"]["date"].min()
    log_info["list_last_create"] = df[df["type"]=="createList"]["date"].max()
    log_info["list_renamed"] = df[df["type"]=="updateList"]["data.old.name"].count() if "data.old.name" in df.columns else 0
    log_info["list_closed"] = df["data.list.closed"].value_counts().to_dict()[True] if "data.list.closed" in df.columns else 0
    log_info["start"] = min(df["date"])
    log_info["ends"] = max(df["date"])
    log_info["first_event_type"] = df.loc[df["date"].argmin(),"type"]
    log_info["members"] = df["idMemberCreator"].nunique()
    log_info["events_per_member"] = df["idMemberCreator"].value_counts().describe().to_dict()
    log_info["card_members_assigned"] = (df["type"]=="addMemberToCard").sum()
    log_info["card_checklists"] = (df["type"]=="addChecklistToCard").sum()
    log_info["card_movement"] = df[df["type"]=="updateCard"]["data.listBefore.id"].count() if "data.listBefore.id" in df.columns else 0
    log_info["card_closed"] = df["data.card.closed"].value_counts().to_dict()[True] if "data.card.closed" in df.columns else 0
    log_info["card_deleted"] = (df["type"]=="deleteCard").sum()
    log_info["card_due"] = df["data.card.due"].count() if "data.card.due" in df.columns else 0
    log_info["types"] = df["type"].value_counts().to_dict()

    if "data.card.id" in df.columns:
        log_info["cards_moving_perc"] = df[~df["data.listBefore.id"].isna()]['data.card.id'].nunique()/log_info["cards"] if "data.listBefore.id" in df.columns else 0
        log_info["cards_checklist_perc"] = df[df["type"]=="addChecklistToCard"]['data.card.id'].nunique()/log_info["cards"] 
        log_info["cards_assigned_perc"] = df[df["type"]=="addMemberToCard"]['data.card.id'].nunique()/log_info["cards"] 
        log_info["cards_closed_perc"] = df[df["data.card.closed"]==True]['data.card.id'].nunique()/log_info["cards"] if "data.card.closed" in df.columns else 0


    return log_info


def list_renames(df):
    return df['data.list.name'].groupby(df['data.list.id']).unique()


def enrich_log(df):
    if not "data.card.closed" in df.columns:
        df["data.card.closed"] = False

    df["data.list.comb"] = (df["data.list.name"]+"["+df["data.list.id"].str.slice(18,24)+"]")
    if df["data.list.id"].str.slice(18,24).nunique() != df["data.list.id"].nunique():
        print("WARNING: list abbreviation is missing information")

    if not "data.listBefore.id" in df.columns:
        df["data.listBefore.id"] = None
    else:
        df["data.listBefore.comb"] = (df["data.listBefore.name"]+"["+df["data.listBefore.id"].str.slice(18,24)+"]")

    if not "data.listAfter.id" in df.columns:
        df["data.listAfter.id"] = None
    else:
        df["data.listAfter.comb"] = (df["data.listAfter.name"]+"["+df["data.listAfter.id"].str.slice(18,24)+"]")

    if not "data.old.name" in df.columns:
        df["data.old.name"] = None

    df.loc[df["type"]=="createList", 'l'] = "list_create"
    df.loc[df["type"]=="moveListToBoard", 'l'] = "list_import"
    df.loc[df["type"].isin(['updateList']), 'l'] = "list_change"
    df.loc[(df["type"]=="updateList") & (~df["data.old.name"].isna()),'l'] = "list_rename"
    df.loc[df["type"].isin(["moveListFromBoard"]), 'l'] = "list_move"
    if "data.list.closed" in df.columns:
        df.loc[df["data.list.closed"].fillna(False), 'l'] = "list_ends" 
    else:
        df["data.list.closed"] = False



    df.loc[card_action_filter(df), 'c'] = "card_act"
    df.loc[card_create_filter(df), 'c'] = "card_create"
    df.loc[card_movement_filter(df), 'c'] = "card_move"
    df.loc[card_movement_filter(df), 'data.list.comb'] = df.loc[card_movement_filter(df), 'data.listBefore.comb']
    df.loc[df["type"]=="deleteCard", 'c'] = "card_delete"
    df.loc[card_closed_filter(df), 'c'] = "card_close"
    

def card_create_filter(df):
    return df["type"].isin(["createCard", "copyCard"])

def card_movement_filter(df):
    return ~df["data.listBefore.id"].isna()

def card_closed_filter(df):
    return df["data.card.closed"].fillna(False)

def card_action_filter(df):
    return df["type"].isin(["updateCard", "addMemberToCard", "commentCard", "addChecklistToCard", "updateCheckItemStateOnCard", "removeMemberFromCard", "updateChecklist", "removeChecklistFromCard", "addAttachmentToCard", "deleteAttachmentFromCard"])

def board_evolution(df, bins=30, list_name=None, all_lists=None):
    bins = pd.cut(df["date"], bins=30)    

    if list_name is not None:
        list_name_filter = df["card.list.name"]==list_name
        df = df[list_name_filter]
        bins = bins[list_name_filter]

    def bin_filter(filter):
        if all_lists is not None:
            groups = [bins[filter], 'data.list.name']
        else:
            groups = bins[filter]
        return df[filter].groupby(groups)['id'].count()

    list_creation_filter = df["type"].isin(["createList", "updateList", "moveListFromBoard", "moveListToBoard"])
    events_filter = ~df['id'].isna()

    cc = {
        'events': bin_filter(events_filter),
        'list_creation': bin_filter(list_creation_filter),
        'card_creation': bin_filter(card_create_filter(df)),
        'card_movement': bin_filter(card_movement_filter(df)),
        'card_closed': bin_filter(card_closed_filter(df)),
        'card_action': bin_filter(card_action_filter(df))
    }

    if all_lists is not None:
        return cc[all_lists]
    else:
        return pd.concat(cc, axis=1)


def list_evolution(df, filter_short_lists=None):
    """
        Returns a dataframe with the begin date, end date, names, and last name of every
        list in the board. Parameter filter_short_lists is a Timedelta that filters those
        lists whose duration is shorter than the value provided.
    """
    list_group = df[~df["l"].isna()].groupby("data.list.id")

    begin_date = list_group['date'].min()
    begin_date.name = "begin_date"

    # For those who haven't finished, the last_date is the last modification in the board
    last_date = list_group['date'].max()
    finished = list_group['l'].first().isin(["list_ends", "list_move"])
    last_date.loc[~finished] = df["date"].max()
    last_date.name = "last_date"

    names = list_group['data.list.name'].unique()
    last_name = list_group['data.list.name'].first()
    last_name.name = "last_name"

    result = pd.concat([begin_date, last_date, last_name, names], axis=1).sort_values('last_date')

    if filter_short_lists is not None:
        return result[(result["last_date"] - result["begin_date"] > filter_short_lists)]
    else:
        return result

def detect_redesign(df, threshold, l_type = None, threshold_l_events = 0):
    df[df["l"].isna()].groupby((~df["l"].isna()).cumsum()[df["l"].isna()])['date'].transform(np.min)

    # The goal of this is to create a Series where the True value represent those events that are part of
    # a redesign. We consider that these events are those that are within a configurable threshold time 
    # distance from a list-related event. We sort the list because we are interested only in those events
    # that occur after a list-related event, not before.
    df_rev = df.sort_index(ascending=False)

    if l_type is None:
        l_events = ~df_rev["l"].isna()
    else:
        l_events = df_rev["l"].isin(l_type)


    if isinstance(threshold, pd.Timedelta):
        r_events = df_rev.groupby(l_events.cumsum())['date'].transform(lambda x: x - np.min(x)) < threshold
        redesign_events = r_events | (~df_rev["l"].isna())
    #b = df.groupby((~df["l"].isna()).cumsum())['date'].transform(lambda x: np.max(x) - x) < threshold
    #redesign_events = (a | b)
    else:
        r_events = (df_rev.groupby(l_events.cumsum())['id'].transform('count') < threshold)
        redesign_events = r_events | (~df_rev["l"].isna())
        # asdf = pd.concat([(df[df["l"].isna()].groupby((~df["l"].isna()).cumsum()[df["l"].isna()])['id'].transform('count') < 15), ~df["l"].isna()], axis=1)
        # c = pd.concat([(asdf["l"] | asdf["id"]), (asdf["l"] | asdf["id"]).shift(1,fill_value=False)], axis=1)
        # pd.concat([(c[0] & ~c[1]).cumsum(), c[0], df['date']], axis=1)
        # df[df["l"].isna()].groupby((~df["l"].isna()).cumsum()[df["l"].isna()])['id'].transform('count')        


    redesign_events.sort_index(ascending=True, inplace=True)

    signal_redesign = redesign_events & ~redesign_events.shift(1, fill_value=False)
    count_l_events = df[~(df["l"].isna())].groupby(signal_redesign.cumsum())['id'].count()    
    count_l_events.name = "count_l_events"

    result = pd.concat([df[redesign_events].groupby(signal_redesign.cumsum())['date'].agg(['min', 'max', 'count']), count_l_events], axis=1)

    if threshold_l_events > 0:
        return result[result["count_l_events"] > threshold_l_events]
    else:
        return result

def plot_list_diagram(list_evolution, begin_end_redesign, ax):
    lists = {p: i for i,p in enumerate(list(list_evolution.index.values))}
    min_date = min(list_evolution['begin_date']) - pd.Timedelta('5D')
    max_date = max(list_evolution['last_date'])

    for index, row in list_evolution.iterrows():
        #print(row['begin_date'], row['last_date'], (lists[index)
        ax.broken_barh([(row['begin_date'], row['last_date']-row['begin_date'])], (lists[index] - 0.45, 0.9), facecolors=plt.cm.plasma(lists[index] / len(lists)))

    ax.vlines(begin_end_redesign["min"], 0, len(lists), colors='tab:red')
    ax.vlines(begin_end_redesign["max"], 0, len(lists), colors='tab:blue')

    ax.set_yticks(range(len(lists)))
    ax.set_yticklabels(list_evolution['last_name'])
    ax.set_ylim(bottom=0, top = len(lists))

    ax.set_xlim(left = min_date, right=max_date)
    ax.set_xlabel("Date", fontdict={'family': 'DejaVu Sans', 'color':  'black', 'weight': 'bold', 'size': 14})
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    ax.tick_params(which='major', axis='x', rotation=90, length=11, color='black')    

def plot_card_actions(df, filter=None, use='comb', **kwargs):
    if filter is None:
        first = df
    elif isinstance(filter, tuple):
        first = df[(df["date"].dt.date > filter[0]) & df["date"].dt.date < filter[1] ]
    else: 
        first = df[(df["date"].dt.date < filter)]

    if use == 'comb':
        y = "data.list.comb"
    elif use == 'id':
        y = "data.list.id"
    else:
        y = "data.list.name"

    chart = sns.displot(first[first["c"].isin(["card_act", "card_create"])], x="date", row="c", binwidth=1, y=y, kind='hist', **kwargs)    
    chart.set_xticklabels(rotation=45)


def to_event_log(df):
    log = df[df["c"].isin(["card_create", "card_close", "card_move"])].copy()
    ll = log.loc[:, ["idMemberCreator", "type", "date", "data.card.closed", "data.card.id", "data.list.name", "data.listAfter.name", "data.listBefore.name"]]
    ll.columns=["org:resource", "type", "time:timestamp", "closed", "case:concept:name", "concept:name", "after", "before"]
    ll.loc[ll["closed"].fillna(False), "concept:name"] = "**Closed"
    ll.loc[~ll["after"].isna(), "concept:name"] = ll["after"]    
    ll["closed"].fillna(False, inplace=True)
    ll["after"].fillna('', inplace = True)
    ll["before"].fillna('', inplace = True)

    lldf = pm4py.format_dataframe(ll.sort_values("time:timestamp"))
    return pm4py.convert_to_event_log(lldf)

def transition_matrix(df, use='names'):
    if use == 'names':
        return df.groupby(['data.listBefore.name', 'data.listAfter.name'])['id'].count().unstack()
    elif use == 'id':
        conversion_map = df.groupby("data.list.id")["data.list.name"].first().to_dict()
        matrix = df.groupby(['data.listBefore.id', 'data.listAfter.id'])['id'].count().unstack()
        return matrix.rename(index=conversion_map, columns=conversion_map)
    else: 
        return df.groupby(['data.listBefore.comb', 'data.listAfter.comb'])['id'].count().unstack()

def connected_lists(df, use='id'):
    G = nx.Graph()
    if use == 'id':
        G.add_nodes_from(df['data.list.id'].dropna())
        G.add_edges_from(df[['data.listAfter.id','data.listBefore.id']].dropna().to_numpy().tolist())
    else:
        G.add_nodes_from(df['data.list.comb'].dropna())
        G.add_edges_from(df[['data.listAfter.comb','data.listBefore.comb']].dropna().to_numpy().tolist())

    return list(nx.connected_components(G))
