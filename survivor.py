import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_sched_grid(schedule_raw):
    sdata = []
    for i, col in enumerate(schedule_raw.columns):
        for matchup in schedule_raw[col].iteritems():
            t1 = matchup[0]
            t2 = matchup[1]
            if t2.find('@') == -1 or t2 == 'BYE':
                continue
            t2 = t2.replace('@', '')
            sdata.append({'week': i + 1, 't1': t1, 't2': t2, 'ht': t2})
    sched = pd.DataFrame(sdata)
    return sched


def get_days_off(sched):
    t1_dates = set(sched.groupby(['t1', 'date']).groups.keys())
    t2_dates = set(sched.groupby(['t2', 'date']).groups.keys())
    t1_dates.update(t2_dates)
    date_df = pd.DataFrame(t1_dates, columns=['team', 'date'])

    days_off = pd.DataFrame([])
    for t in date_df.team.unique():
        tdates = date_df[date_df.team == t].sort_values('date')
        tdates['days_off'] = tdates.date - tdates.date.shift(1)
        tdates['days_off'] = tdates.days_off.dt.days.fillna(7)
        days_off = pd.concat([days_off, tdates])

    sched = pd.merge(sched, days_off, left_on=['t1', 'date'],
                     right_on=['team', 'date'], how='left')
    sched.rename(columns={'days_off': 't1_days_off'}, inplace=True)
    sched.drop(columns=['team'], inplace=True)

    sched = pd.merge(sched, days_off, left_on=['t2', 'date'],
                     right_on=['team', 'date'], how='left')
    sched.rename(columns={'days_off': 't2_days_off'}, inplace=True)
    sched.drop(columns=['team'], inplace=True)
    return sched


def prep_data(inp_data, HOME_BOOST = 1, SHORT_WEEK_PENATLY = 1,
              WORST_TEAMS_PENALTY = .3, INTL_PRIOR_PENALTY = .5):
    data = inp_data.copy()
    # Boost home team, hurt 4 day turn arounds, and prior intl
    data['t2_fpi'] += HOME_BOOST
    data['t2_elo'] += HOME_BOOST

    data.loc[data.t1_days_off <= 6, 't1_fpi'] -= SHORT_WEEK_PENATLY
    data.loc[data.t1_days_off <= 6, 't1_elo'] -= SHORT_WEEK_PENATLY
    data.loc[data.t2_days_off <= 6, 't2_fpi'] -= SHORT_WEEK_PENATLY
    data.loc[data.t2_days_off <= 6, 't2_elo'] -= SHORT_WEEK_PENATLY

    data.loc[data.t1 == data.intl_prior, 't1_fpi'] -= INTL_PRIOR_PENALTY
    data.loc[data.t1 == data.intl_prior, 't1_elo'] -= INTL_PRIOR_PENALTY
    data.loc[data.t2 == data.intl_prior, 't2_fpi'] -= INTL_PRIOR_PENALTY
    data.loc[data.t2 == data.intl_prior, 't2_elo'] -= INTL_PRIOR_PENALTY

    # Calculate diffs and  final score
    data['elo_diff'] = data.t1_elo - data.t2_elo
    data['fpi_diff'] = data.t1_fpi - data.t2_fpi
    std_scaler = StandardScaler()
    data['days_off_diff'] = std_scaler.fit_transform(data[['days_off_diff']])

    data['score'] = data['elo_diff'] + data['days_off_diff'] + data['fpi_diff']

    data.loc[(data.score > 0) & (data.t1_rank >= 28), 'score'] -= WORST_TEAMS_PENALTY
    data.loc[(data.score < 0) & (data.t2_rank >= 28), 'score'] -= WORST_TEAMS_PENALTY

    data = data.loc[~((data.week >= 17) & ((data.t1_rank <=5) | (data.t2_rank <=6)))]
    return data.reset_index(drop=True)


def simulate_greedy(inp_data, max_col='score', week_max=18,
                    picked_teams_inp=[], bottom_up=False):
    double_weeks = list(range(14, 19))
    sdata = inp_data.copy()
    sdata['abs_score'] = abs(sdata[max_col])
    picked_weeks = list(range(1, len(picked_teams_inp) + 1))
    start = len(picked_weeks)
    end = week_max + len(double_weeks)

    output_weeks = picked_weeks.copy()
    output_teams = picked_teams_inp.copy()
    output_scores = [10] * len(picked_teams_inp)

    for rd in range(start, end):
        rd_data = sdata.copy()
        rd_data = rd_data[(rd_data.week <= week_max)]
        rd_data = rd_data[~((rd_data.t1.isin(output_teams)) &
                            (rd_data.score > 0))]
        rd_data = rd_data[~((rd_data.t2.isin(output_teams)) &
                            (rd_data.score < 0))]
        rd_data = rd_data[~rd_data.week.isin(picked_weeks)]

        if len(rd_data) == 0:
            continue

        grp = rd_data.groupby('week')
        week_scores = grp['abs_score'].max().sort_values(ascending=bottom_up)

        pick_wk = week_scores.index[0]
        pick_diff = week_scores[pick_wk]
        pick_row = rd_data[(rd_data.week == pick_wk) &
                           (rd_data['abs_score'] == pick_diff)]
        pick = pick_row.t1.iloc[0] if pick_row.score.iloc[0] > 0 else pick_row.t2.iloc[0]

        if pick_wk in double_weeks:
            double_weeks.remove(pick_wk)
        else:
            picked_weeks.append(pick_wk)
        output_weeks.append(pick_wk)
        output_teams.append(pick)
        output_scores.append(pick_diff)

    pick_df = pd.DataFrame(data={'week': output_weeks,
                                 'team': output_teams,
                                 'score': output_scores})
    return pick_df.sort_values('week').reset_index(drop=True)


def simulate_random(inp_data, max_col='score', week_max=18,
                    picked_teams_inp=[], top_n=3):
    double_weeks = list(range(14, 19))
    sdata = inp_data.copy()
    sdata['abs_score'] = abs(sdata[max_col])
    picked_weeks = list(range(1, len(picked_teams_inp) + 1))
    start = len(picked_weeks)
    end = week_max + len(double_weeks)

    output_weeks = picked_weeks.copy()
    output_teams = picked_teams_inp.copy()
    output_scores = [10] * len(picked_teams_inp)

    for rd in range(start, end):
        rd_data = sdata.copy()
        rd_data = rd_data[(rd_data.week <= week_max)]
        rd_data = rd_data[~((rd_data.t1.isin(output_teams)) &
                            (rd_data.score > 0))]
        rd_data = rd_data[~((rd_data.t2.isin(output_teams)) &
                            (rd_data.score < 0))]
        rd_data = rd_data[~rd_data.week.isin(picked_weeks)]

        if len(rd_data.week.unique()) == 1:
            pick_wk = rd_data.week.iloc[0]
            pick_diff = rd_data.abs_score.max()
            pick_row = rd_data[(rd_data.week == pick_wk) &
                               (rd_data['abs_score'] == pick_diff)]
        else:
            grp = rd_data.groupby('week')
            week_scores = grp['abs_score'].nlargest(top_n).reset_index(level=1, drop=True)
            random_pick = week_scores.sample(1)
            pick_wk = random_pick.index[0]
            pick_diff = random_pick[pick_wk]
            pick_row = rd_data[(rd_data.week == pick_wk) &
                               (rd_data['abs_score'] == pick_diff)]
        pick = pick_row.t1.iloc[0] if pick_row.score.iloc[0] > 0 else pick_row.t2.iloc[0]

        if pick_wk in double_weeks:
            double_weeks.remove(pick_wk)
        else:
            picked_weeks.append(pick_wk)
        output_weeks.append(pick_wk)
        output_teams.append(pick)
        output_scores.append(pick_diff)

    pick_df = pd.DataFrame(data={'week': output_weeks,
                                 'team': output_teams,
                                 'score': output_scores})
    return pick_df.sort_values('week').reset_index(drop=True)


def main():
    import argparse

    ##################################
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-f', '--format_schedule', action='store_true', help='Format raw schedule')
    arg_parser.add_argument(
        '-s', '--simulate', action='store_true', help='Simulate best selections')
    args = arg_parser.parse_args()
    ##################################

    if args.format_schedule:
        schedule_grid_path = '~/repos/nfl-survivor/data/schedule_grid.csv'
        formatted_output_path = '~/repos/nfl-survivor/data/schedule_processed.csv'
        schedule_raw = pd.read_csv(schedule_grid_path, index_col=0)
        formatted_schedule = process_sched_grid(schedule_raw)
        formatted_schedule.to_csv(formatted_output_path)
    elif args.simulate:
        formatted_schedule_path = '~/repos/nfl-survivor/data/schedule_processed.csv'
        rank_path = '~/repos/nfl-survivor/data/ranks.csv'

        sched = pd.read_csv(formatted_schedule_path, parse_dates=['date'])
        sched = get_days_off(sched)
        sched['days_off_diff'] = sched.t1_days_off - sched.t2_days_off

        std_scaler = StandardScaler()
        mm_scaler = MinMaxScaler()

        rankings = pd.read_csv(rank_path)
        rankings['elo_scaled'] = std_scaler.fit_transform(rankings[['elo']])
        rankings['fpi_scaled'] = std_scaler.fit_transform(rankings[['fpi']])

        data = pd.merge(sched, rankings[['team', 'rank', 'elo_scaled', 'fpi_scaled']],
                        left_on='t1',
                        right_on='team')
        data.rename(columns={'elo_scaled': 't1_elo', 'fpi_scaled': 't1_fpi',
                             'rank': 't1_rank'},
                    inplace=True)
        data.drop(columns=['team'], inplace=True)

        data = pd.merge(data, rankings[['team', 'rank', 'elo_scaled', 'fpi_scaled']],
                        left_on='t2',
                        right_on='team')
        data.rename(columns={'elo_scaled': 't2_elo', 'fpi_scaled': 't2_fpi',
                             'rank': 't2_rank'},
                    inplace=True)
        data.drop(columns=['team'], inplace=True)

        HOME_BOOST = .75
        SHORT_WEEK_PENATLY = .66
        WORST_TEAMS_PENALTY = .4
        INTL_PRIOR_PENALTY = .5

        sim_data = prep_data(data, HOME_BOOST, SHORT_WEEK_PENATLY,
                             WORST_TEAMS_PENALTY, INTL_PRIOR_PENALTY)

        m1_picks = []

        greedy = simulate_greedy(sim_data, 'score', picked_teams_inp=m1_picks,
                                 bottom_up=False)

        bu_greedy = simulate_greedy(sim_data, 'score', picked_teams_inp=m1_picks,
                                    bottom_up=True)

        sims = {}
        for i in tqdm(range(10000)):
            try:
                picks = simulate_random(sim_data, 'score', top_n=3,
                                        picked_teams_inp=m1_picks)
            except:
                continue
            if picks.score.sum() > 109:
                sims[picks.score.sum()] = picks

        import ipdb; ipdb.set_trace(context=25)
        print(greedy)
        print(bu_greedy)
        print(sims[0])


if __name__ == '__main__':
    main()
