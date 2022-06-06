import pickle
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from catboost import CatBoostClassifier

##方差筛选
def variance_selection(feature_frame):
    var_selector = VarianceThreshold(threshold = 0)#设置方差过滤阈值为1
    var_selector.fit_transform(feature_frame)
    is_select = var_selector.get_support()#get_support函数返回方差＞阈值的布尔值序列
    return is_select

def sample(df, sampled=False, frac=0.2):
    if sampled:
    # Split click bools
        click_msk = df['click_bool'] == 1
        df_click = df[click_msk]
        df_noclick = df[~click_msk]

        # Stratified sampling
        df_sampled = df_noclick.groupby('srch_id').sample(frac=frac, replace=False, random_state=23)
        df_sampled = pd.concat([df_sampled, df_click], ignore_index=True)

        df_sampled.sort_values(by=['srch_id'], inplace=True)
        return df_sampled
    else:
        return df

##相关性筛选
def feature_selection(feature_frame, label):
    mutualInfo = mutual_info_regression(feature_frame, label, discrete_features='auto')
    mutualInfo_select = pd.Series(data = mutualInfo , index = feature_frame.columns).sort_values(ascending = False)
    feature_frame.insert(len(feature_frame.columns),'label',label)#再添加一列
    corr = feature_frame.corr()
    corr_series = pd.Series(data = corr.iloc[-1,:-1])
    corr_sort = corr_series.abs().sort_values(ascending = False)
    plt.figure(figsize=(20,10))
    sns.barplot(mutualInfo_select.values[:49], mutualInfo_select.index[:49], orient='h')
    plt.title("Correlation screening based on mutual information")
    plt.savefig('images/feature_mutual_info.png')
    plt.show()

    plt.figure(figsize=(20,10))
    sns.barplot(corr_sort.values[:49], corr_sort.index[:49], orient='h')
    plt.title("Correlation screening based on correlation coefficient")
    plt.savefig('images/feature_corrlation.png')
    plt.show()
    return mutualInfo_select.index[:49].to_list(), corr_sort.index[:49].to_list()


def add_feature(df_fe): 
    df_fe.insert(len(df_fe.columns) - 1, 'click_bool', df_fe.pop('click_bool'))
    '''
    Add feature engineered features to dataframes
    '''
    df_hotel = pd.read_csv('data/hotel_data.csv')
    df_fe = df_fe.join(df_hotel.set_index('prop_id'), on='prop_id', rsuffix='_cnt')

    feat_list = ['price_usd_mean_ctry']
    prop_avg_price_country = pd.read_csv('data/prop_avg_price_country.csv')
    df_fe = df_fe.join(prop_avg_price_country.set_index('prop_country_id')[feat_list], on='prop_country_id')

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
    #price rank The price rank of this hotel within one search
    for feat in  ['price_usd', 'prop_location_score2']:
        df_fe[f'{feat}_rank'] = df_fe.groupby('srch_id')[feat].rank('dense', ascending=False)
    
    df_fe['comp_inv'] = df_fe['comp2_inv'] + df_fe['comp3_inv'] + df_fe['comp5_inv'] + df_fe['comp8_inv']
    df_fe['comp_rate'] = df_fe['comp2_rate'] + df_fe['comp3_rate'] + df_fe['comp5_rate'] + df_fe['comp8_rate']
    df_fe.drop(columns=['comp2_rate', 'comp2_inv', 'comp3_rate', 'comp3_inv', 'comp5_rate', 'comp5_inv', 'comp8_rate', 'comp8_inv', \
        'agg_prop_location_score1_mean', 'agg_prop_location_score1_std', 'agg_prop_location_score1_median', 'agg_prop_starrating_mean', 'agg_prop_starrating_std', \
        'agg_prop_review_score_mean', 'agg_prop_review_score_std', 'agg_prop_review_score_median', 'agg_prop_starrating_median'], inplace=True)

    df_fe.fillna(0, inplace=True)

    return df_fe

def NDCG(ordered, n=5):
    """normalised discounted cumulative gain

    Parameters
    ----------
    ordered_data : 1D numpy array, the list of relevance judgments which takes 3 values separately: 
        0(no clicked), 1(clicked), 5(booked), the list sequence is based on the prediction probability
        of the model(e.g. XGboost) 
        For exampe: if the predicted result is [0.1, 0.3, 0.4, 0.5, 0.2, 0.4]
                    the correct label is [5, 0, 1, 5, 1, 0], 
                    we can use ordered = [x for _,x in sorted(zip(np.array(predicted),np.array(labled)), reverse=True)]
                    to get the ordered [5, 1, 0, 0, 1, 5]
    p : int, the length of ordered_data

    Returns
    ----------
    ndcg : float, the ndcg evaluation for the ordered_data (NDCG=DCG/IDCG)  
        in the above example, the result is 0.83629101594232
    """
    if sum(ordered)==0:
        return 0
    else:
        index = list(range(n))
        DCG = np.sum((2**np.array(ordered)-1)[:n]/np.log2(np.array(index)+2))
        IDCG = np.sum((2**np.array(sorted(ordered)[::-1])-1)[:n]/np.log2(np.array(index)+2)) 
        return DCG/IDCG

def plot_feature_importance(model_importances, method):
    fig, ax = plt.subplots(figsize=(20,10))
    model_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    for bars in ax.containers:
        ax.bar_label(bars)
    ax.set_axisbelow(True)
    ax.grid(which='major')
    fig.tight_layout()
    plt.savefig(f'images/feature_importance_{method}_inter.png')
    plt.show()

def main(method, name):
    pd.options.mode.chained_assignment = None
    start_time = time.time()

    print('--- Load Data ---')
    
    # df_train = pd.read_csv('data/train_cleaned_final.csv').iloc[:,2:]
    # print('--- Add Feature ---')
    # df_fe = add_feature(df_train)
    # df_fe.to_csv('data/train_all_feature.csv')
    df_fe = pd.read_csv('data/train_all_feature.csv')
    df_fm = pd.read_csv('data/fm_rank.csv')
    df_fe = df_fe.merge(df_fm[['srch_id', 'prop_id', 'fm_rank']], on=['srch_id', 'prop_id'],  how='left')
    print(df_fe.fm_rank.isnull().sum()*100/df_fe.shape[0])
    df_fe.fillna(0, inplace=True)

    '''
    Drop noise
    '''
    count = df_fe.groupby('srch_id')["srch_id"].count().to_frame('group_cnt')
    df_fe = df_fe.join(count, on='srch_id')
    df_fe = df_fe[~(df_fe.group_cnt < 5)]

    '''
    Add Label
    '''
    df_fe = df_fe
    book_msk = df_fe['booking_bool'] == 1
    df_fe['label'] = df_fe['click_bool']
    df_fe['label'][book_msk] = 2

    print('--- Train test split ---')

    '''
    Balance dataset here. 
    '''
    # Stratified train/test split
    strata = np.unique(df_fe['srch_id'])
    strata_train = np.random.choice(strata, round(len(strata) * 0.7), replace=False)
    train_msk = df_fe['srch_id'].isin(strata_train)
    df_fe_train = df_fe[train_msk]
    df_fe_temp = df_fe[~train_msk]

    # Stratified valid/test split
    strata = np.unique(df_fe_temp['srch_id'])
    strata_test = np.random.choice(strata, round(len(strata) * 0.3), replace=False)
    test_msk = df_fe_temp['srch_id'].isin(strata_test)
    df_fe_val = df_fe_temp[test_msk]
    df_fe_test = df_fe_temp[~test_msk]

    df_sampled = sample(df_fe_train)
    # df_sampled_fe = sample(df_fe_train, True, 0.2)
    
    print(f'original shape: {df_fe_train.shape}, sampled shape: {df_sampled.shape}')

    group_train = df_sampled.groupby("srch_id")["srch_id"].count().to_numpy()
    group_val = df_fe_val.groupby("srch_id")["srch_id"].count().to_numpy()
    group_test = df_fe_test.groupby("srch_id")["srch_id"].count().to_numpy()
    df_label_train = df_sampled.iloc[:, -1]
    # df_label_train_fe = df_sampled_fe.iloc[:, -1]
    y_train = df_label_train.to_numpy()
    y_val = df_fe_val.iloc[:, -1].to_numpy()
    y_test = df_fe_test.iloc[:, -1].to_numpy()

    drop_list = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', \
        'Unnamed: 0', 'booking_bool', 'click_bool', 'position',  'prop_id', 'srch_destination_id', 'label', 'group_cnt']


    print('--- Feature selection ---')
    # df_sampled_fe.drop(drop_list, axis=1, inplace=True)
    # feat_list1, feat_list2 = feature_selection(df_sampled_fe, df_label_train_fe)
    # feat_list3 = ['price_usd_rela_diff', 'ump', 'click_prob', 'orig_destination_distance', 'per_fee', 'date_time_unix', 'prop_location_score2_rela_diff', \
    #     'prop_location_score2_abso_diff', 'price_usd_abso_diff', 'srch_hist_timepoint', 'day', 'hour', 'srch_destination_id_cnt', 'srch_booking_window']
    # all_feat = list(set(feat_list1).union(set(feat_list2)))
    # all_feat = list(set(all_feat).union(set(feat_list3)))

    all_feat = pickle.load(open("feature.txt", "rb"))
    feat_list4 = ['prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', \
        'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', \
        'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'orig_destination_distance', 'random_bool', 'srch_hist_bool', \
        'srch_count', 'promotion_flag_cnt', 'random_bool_cnt', 'position_mean', 'position_std', 'position_median', 'agg_prop_location_score2_mean', \
        'agg_prop_location_score2_median', 'agg_prop_location_score2_std', 'agg_price_usd_mean', 'agg_price_usd_median', 'agg_price_usd_std', \
        'srch_hist_timepoint', 'booking_bool_cnt', 'click_bool_cnt', 'srch_query_affinity_score', 'srch_query_affinity_score_median', 'price_usd_mean_ctry', \
        'date_time_unix', 'month', 'day', 'hour', 'ump', 'count_window', 'price_diff', 'starrating_diff', 'total_fee', 'per_fee', \
        'people_per_room', 'score1d2', 'score2ma', 'country_price_diff', 'book_prob1', 'srch_destination_id_cnt', 'price_usd_mean_dest', \
        'site_country_common_bool', 'price_usd_abso_diff', 'prop_starrating_abso_diff', 'prop_review_score_abso_diff', \
        'prop_location_score1_abso_diff', 'prop_location_score2_abso_diff', 'price_usd_rela_diff', 'prop_starrating_rela_diff', \
        'prop_review_score_rela_diff', 'prop_location_score1_rela_diff', 'prop_location_score2_rela_diff', 'click_prob', 'fm_rank'] 
    all_feat = feat_list4

    # all_feat = list(set(all_feat).intersection(set(feat_list4)))
    # df_sampled.drop(drop_list, axis=1, inplace=True)
    # all_feat = df_sampled.columns.to_list()


    print(f'new sampled shape: {df_sampled[all_feat].shape}')
    X_train = df_sampled[all_feat].to_numpy()
    X_val = df_fe_val[all_feat].to_numpy()
    X_test = df_fe_test[all_feat].to_numpy()

    print(f'All feature : {all_feat}')
    with open(f'feature_{name}.txt', 'wb') as text:
        pickle.dump(all_feat, text)
    print()
    print('--- Run model ---')

    if method == 'gbm':
        gbm = lgb.LGBMRanker(objective='lambdarank', num_leaves=32, max_depth=12, metric='ndcg', learning_rate=0.05, n_estimators=3000,
                                    verbosity=2, force_col_wise=True)
        gbm.fit(X_train, y_train, group=group_train,
                eval_set=[(X_val, y_val)], eval_group=[group_val],
                eval_at=[1, 5])

        print('--- Plot feature importance ---')
        y_labels = np.flip(np.array(df_sampled[all_feat].columns)[gbm.feature_importances_.argsort()])
        result = np.flip(np.sort(gbm.feature_importances_))
        gbm_importances = pd.Series(result, index=y_labels)
        plot_feature_importance(gbm_importances, method)

        print(f'--- Feature Number: {X_train.shape[1]} ---')
        print()
        print('--- Predict test ---')
        test_pred = gbm.predict(X_test)
        model = gbm

    elif method == 'cat':
        cat = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            task_type='GPU',
            loss_function='MultiClass',
        #     gpu_ram_part=0.9,
        #     boosting_type='Plain',
        #     max_ctr_complexity=2,
        #     depth=6,
        #     gpu_cat_features_storage='CpuPinnedMemory',
        )
        # cat_features = [range(X_train.shape[1])]
        cat.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=None,
            verbose=10,
        )
        print(cat.classes_)
        y_labels = np.flip(np.array(df_sampled[all_feat].columns)[cat.get_feature_importance(prettified=False).argsort()])
        result = np.flip(np.sort(cat.get_feature_importance(prettified=False)))
        cat_importances = pd.Series(result, index=y_labels)
        plot_feature_importance(cat_importances, method)
        print('--- Predict test ---')
        test_pred = cat.predict(X_test, 
                        prediction_type='Probability', 
                        ntree_start=0, ntree_end=cat.get_best_iteration(), 
                        thread_count=-1, verbose=None)
        print(f'predict shape{test_pred.shape}')
        test_pred = test_pred[:,2]
        model = cat
    
    elif method == 'randomforest':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = RandomForestClassifier(criterion='entropy', max_depth=10, n_jobs=-1, verbose=2, random_state = 42, n_estimators=500)
        clf.fit(X_train, y_train)
        print(clf.classes_)
        y_labels = np.flip(np.array(df_sampled[all_feat].columns)[clf.feature_importances_.argsort()])
        result = np.flip(np.sort(clf.feature_importances_))
        rf_importances = pd.Series(result, index=y_labels)
        plot_feature_importance(rf_importances, method)
        print('--- Predict test ---')
        test_pred = clf.predict_proba(X_test)
        test_pred = test_pred[:,-1]
        model = clf


    ## compute ndgc for each query.
    nquerys=range(0,len(group_test))
    lower=0
    upper=0
    ndcgs_5=[]
    ndcgs_1=[]
    for i in nquerys:
            many=group_test[i]
            upper = upper+many
            predicted = test_pred[lower:upper]
            labled = y_test[lower:upper]
            ordered = [x for _,x in sorted(zip(predicted,labled), reverse=True)]
            result_1 = NDCG(ordered, 1)
            ndcgs_1.append(result_1)
            try:
                result_5 = NDCG(ordered, 5)
                ndcgs_5.append(result_5)
            except:
                print(f'Exception: NDCG_5 WRONG {len(ordered)}')
                lower=upper
            lower=upper

    print('--- Test Result ---')
    print(f'ndcgs_1: {np.mean(ndcgs_1)}, ndcgs_5: {np.mean(ndcgs_5)}')

    print('--- Save model ---')
    # Save model
    pickle.dump(model, open(f"model/{method}_final_{name}.pkl", "wb"))
    
    print(f'All feature : {all_feat}')
    
    print('--- Finish! Runtime: %s seconds---'%(time.time()-start_time))

if __name__ == "__main__":
    main('gbm', 'inter')