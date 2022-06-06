import pickle
import time
import pandas as pd
import numpy as np
import lightgbm as lgb

from matplotlib import pyplot as plt

def add_feature_test(df_fe):
    df_hotel = pd.read_csv('data/hotel_data.csv')
    df_fe = df_fe.join(df_hotel.set_index('prop_id'), on='prop_id', rsuffix='_cnt')

    feat_list = ['price_usd_mean_ctry']
    prop_avg_price_country = pd.read_csv('data/prop_avg_price_country.csv')
    df_fe = df_fe.join(prop_avg_price_country.set_index('prop_country_id')[feat_list], on='prop_country_id')
    
    # df_fe.insert(len(df_fe.columns) - 1, 'click_bool', df_fe.pop('click_bool'))
   # convert data_time to unix
    df_fe['date_time_unix'] = (pd.to_datetime(df_fe['date_time']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    df_fe['date_time'] = pd.to_datetime(df_fe.date_time, format='%Y-%m-%d %H:%M:%S')
    # df_fe['year'] = df_fe.date_time.dt.year
    df_fe['month'] = df_fe.date_time.dt.month
    df_fe['day'] = df_fe.date_time.dt.day
    df_fe['hour'] = df_fe.date_time.dt.hour
    df_fe['dayofweek'] = df_fe.date_time.dt.dayofweek
    date_list = ['date_time_unix']
    for i in date_list:
        df_srch_hist = pd.read_csv(f'data/srch_hist_{i}.csv')
        df_fe = pd.merge(df_fe, df_srch_hist, on=['prop_id', i], how='left')

    df_fe['ump'] = np.exp(df_fe['prop_log_historical_price']) - df_fe['price_usd']
    df_fe['count_window'] = df_fe['srch_room_count']*df_fe['srch_booking_window'] + df_fe['srch_booking_window']
    df_fe['price_diff'] = df_fe['visitor_hist_adr_usd'] - df_fe['price_usd']
    df_fe['starrating_diff'] = df_fe['visitor_hist_starrating'] - df_fe['prop_starrating']
    df_fe['total_fee'] =  df_fe['price_usd']*df_fe['srch_room_count']
    df_fe['total_people'] = df_fe['srch_adults_count'] + df_fe['srch_children_count']
    df_fe['per_fee'] = df_fe['total_fee']/df_fe['total_people']
    df_fe['people_per_room'] = df_fe['srch_room_count']/df_fe['total_people']
    df_fe['score1d2'] = (df_fe['prop_location_score2'] + 0.0001)/(df_fe['prop_location_score1'] + 0.0001)
    df_fe['score2ma'] = df_fe['prop_location_score2']*df_fe['srch_query_affinity_score']
    df_fe['country_price_diff'] = df_fe['price_usd_mean_ctry'] - df_fe['price_usd']
    
    feat_list = ['price_usd_mean_dest', 'price_usd_std_dest', 'price_usd_median_dest', 'srch_destination_id_cnt']
    df_srch_des = pd.read_csv('data/srch_des_cnt.csv')
    df_fe = df_fe.join(df_srch_des.set_index('srch_destination_id')[feat_list], on='srch_destination_id')

    site_country_id = pd.read_csv('data/site_country_common_id.csv')
    check_site_list = site_country_id[['visitor_location_country_id', 'site_id']].astype('str').agg('&'.join, axis=1).to_list()
    df_fe['check_site_country'] = df_fe[['visitor_location_country_id', 'site_id']].astype('str').agg('&'.join, axis=1)
    df_fe['site_country_common_bool'] = 0
    df_fe['site_country_common_bool'][df_fe.check_site_country.isin(check_site_list)] = 1
    df_fe.drop(columns=['check_site_country'], inplace=True)

    feat_select = ['price_usd', 'prop_starrating' ,'prop_review_score', 'prop_location_score1', 'prop_location_score2']
    # Compute mean per srch_id
    agg_feat = df_fe.groupby('srch_id')[feat_select].agg('mean')
    agg_srch_id = df_fe['srch_id'].to_frame('srch_id').join(agg_feat, on='srch_id') # Expand back to srch_id 
    agg_srch_id.drop('srch_id', axis=1, inplace=True)
    # Absolute difference: Subtract mean
    df_abso_diff = df_fe[feat_select] - agg_srch_id 
    # Relative difference
    df_rela_diff = (df_fe[feat_select] - agg_srch_id)*2 / (np.abs(df_fe[feat_select]) + np.abs(agg_srch_id) + 0.0001)
    # Column renaming 
    df_abso_diff = df_abso_diff.add_suffix('_abso_diff')
    df_rela_diff = df_rela_diff.add_suffix('_rela_diff')
    df_abso_diff.insert(0, 'srch_id', df_fe['srch_id']) 
    df_rela_diff.insert(0, 'srch_id', df_fe['srch_id']) 
    diff_abso_list = ['price_usd_abso_diff', 'prop_starrating_abso_diff', 'prop_review_score_abso_diff', 'prop_location_score1_abso_diff', 'prop_location_score2_abso_diff']
    diff_rela_list = ['price_usd_rela_diff', 'prop_starrating_rela_diff', 'prop_review_score_rela_diff', 'prop_location_score1_rela_diff', 'prop_location_score2_rela_diff']
    df_fe.loc[:, diff_abso_list] = df_abso_diff
    df_fe.loc[:, diff_rela_list] = df_rela_diff
    for feat in ['price_usd', 'prop_location_score2']:
        df_fe[f'{feat}_rank'] = df_fe.groupby('srch_id')[feat].rank('dense', ascending=False)
    
    df_fe['comp_inv'] = df_fe['comp2_inv'] + df_fe['comp3_inv'] + df_fe['comp5_inv'] + df_fe['comp8_inv']
    df_fe['comp_rate'] = df_fe['comp2_rate'] + df_fe['comp3_rate'] + df_fe['comp5_rate'] + df_fe['comp8_rate']
    df_fe.drop(columns=['comp2_rate', 'comp2_inv', 'comp3_rate', 'comp3_inv', 'comp5_rate', 'comp5_inv', 'comp8_rate', 'comp8_inv', \
        'agg_prop_location_score1_mean', 'agg_prop_location_score1_std', 'agg_prop_location_score1_median', 'agg_prop_starrating_mean', 'agg_prop_starrating_std', \
        'agg_prop_review_score_mean', 'agg_prop_review_score_std', 'agg_prop_review_score_median', 'agg_prop_starrating_median'], inplace=True)

        
    return df_fe

def main():
    name = 'inter'
    start_time = time.time()

    print('--- Load Data ---')

    pd.options.mode.chained_assignment = None
    df_test = pd.read_csv('data/test_cleaned_final.csv')

    print(df_test.shape)

    df_fe_test = add_feature_test(df_test) 
    df_fm = pd.read_csv('data/fm_rank.csv')
    df_fe_test = df_fe_test.merge(df_fm[['srch_id', 'prop_id', 'fm_rank']], on=['srch_id', 'prop_id'],  how='left')
    print(df_fe_test.fm_rank.isnull().sum()*100/df_fe_test.shape[0])
    # df_fe_test.fillna(0, inplace=True)

    for col in df_fe_test.columns:
        if df_fe_test[col].isna().any():
            print(f'fillna in the {col}')
            df_fe_test[col] = df_fe_test[col].fillna(0)
    
    feat_list = pickle.load(open(f"feature_{name}.txt", "rb"))
    # feat_list = ['prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', \
    #     'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', \
    #     'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'orig_destination_distance', 'random_bool', 'srch_hist_bool', \
    #     'srch_count', 'promotion_flag_cnt', 'random_bool_cnt','position_mean', 'position_std', 'position_median' 'agg_prop_location_score2_mean', \
    #     'agg_prop_location_score2_median', 'agg_prop_location_score2_std', 'agg_price_usd_mean', 'agg_price_usd_median', 'agg_price_usd_std', \
    #     'srch_hist_timepoint', 'booking_bool_cnt', 'click_bool_cnt', 'srch_query_affinity_score', 'srch_query_affinity_score_median', 'price_usd_mean_ctry', \
    #     'date_time_unix', 'month', 'day', 'hour', 'ump', 'count_window', 'price_diff', 'starrating_diff', 'total_fee', 'per_fee', \
    #     'people_per_room', 'score1d2', 'score2ma', 'country_price_diff', 'book_prob1', 'srch_destination_id_cnt', 'price_usd_mean_dest', \
    #     'site_country_common_bool', 'price_usd_abso_diff', 'prop_starrating_abso_diff', 'prop_review_score_abso_diff', \
    #     'prop_location_score1_abso_diff', 'prop_location_score2_abso_diff', 'price_usd_rela_diff', 'prop_starrating_rela_diff', \
    #     'prop_review_score_rela_diff', 'prop_location_score1_rela_diff', 'prop_location_score2_rela_diff']
    feat_list = ['click_bool' if i =='click_bool_cnt' else i for i in feat_list]
    feat_list = ['booking_bool' if i =='booking_bool_cnt' else i for i in feat_list]
    df_fe = df_fe_test[feat_list]

    # drop_list = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', \
    #     'Unnamed: 0', 'booking_bool', 'click_bool', 'position',  'prop_id', 'srch_destination_id']
    # test_drop_list = [ele for ele in drop_list if ele not in ['position', 'click_bool', 'booking_bool']]
    # df_fe_test.drop(test_drop_list, axis=1, inplace=True)
    print(df_fe.columns)
    print(df_fe.shape)

    gbm = pickle.load(open(f"model/gbm_final_{name}.pkl", "rb"))
    gbm_pred = gbm.predict(df_fe.to_numpy())

    ''' output for test'''
    df_test['score'] = gbm_pred
    rf_pred = df_test[['srch_id', 'prop_id', 'score']].sort_values(['srch_id', 'score'], ascending=[True, False])
    rf_pred[['srch_id', 'prop_id']].to_csv(f'predictions/lgbm_pred_fe_{name}_all.csv', index=False)

    print('--- Finish! Runtime: %s seconds---'%(time.time()-start_time))

if __name__ == "__main__":
    main()