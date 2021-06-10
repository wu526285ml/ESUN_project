import numpy as np    # 計算向量、矩陣的套件
import pandas as pd   # 運用表格 DataFrame的套件
from tqdm import trange   # 顯示進度條
import matplotlib.pyplot as plt  # 畫圖工具


# loading all csv data
# 這邊載入csv檔案，有稍微把 index和一些 column重新命名，如果跳出錯誤可能是因為這些csv檔案並沒有和現在這個python檔案在同一個資料夾

# 客戶資訊、基本屬性資料
# cust_info = pd.read_csv('cust_info.csv',names=['CUST_NO','AGE','OPEN_ACCT','SOURCE','BREACH','BREACH_DATE','BREACH_RANK','TXN_YEAR','B_COUNT','S_COUNT','N_COUNT'])
cust_info = pd.read_csv('cust_info.csv')
# 股票類別
stock_category = pd.read_csv('stock_category.csv',index_col=0,names=['STOCK_NO','INDUSTRY'])
# 股票歷史資料
# stock_info = pd.read_csv('stock_info.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
# stock_info_Q9 = pd.read_csv('stock_info_2021Q1.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
# 每一季客戶交易資訊
txn_Q1 = pd.read_csv('txn_2019Q1.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q2 = pd.read_csv('txn_2019Q2.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q3 = pd.read_csv('txn_2019Q3.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q4 = pd.read_csv('txn_2019Q4.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q5 = pd.read_csv('txn_2020Q1.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q6 = pd.read_csv('txn_2020Q2.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q7 = pd.read_csv('txn_2020Q3.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q8 = pd.read_csv('txn_2020Q4.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])
txn_Q9 = pd.read_csv('txn_2021Q1.csv',index_col=1,names=['DATE','CUST_NO','BS','MARKET','STOCK_NO','COMMISION','PRICE','STOCKS','ROI'])

# concat all txn in a csv
txn = pd.concat([txn_Q1,txn_Q2,txn_Q3,txn_Q4,txn_Q5,txn_Q6,txn_Q7,txn_Q8,txn_Q9],axis=0)



# 先把交易量算出來 (stocks*price)
txn_Q1['TOTAL'] = txn_Q1['PRICE']*txn_Q1['STOCKS']
txn_Q2['TOTAL'] = txn_Q2['PRICE']*txn_Q2['STOCKS']
txn_Q3['TOTAL'] = txn_Q3['PRICE']*txn_Q3['STOCKS']
txn_Q4['TOTAL'] = txn_Q4['PRICE']*txn_Q4['STOCKS']
txn_Q5['TOTAL'] = txn_Q5['PRICE']*txn_Q5['STOCKS']
txn_Q6['TOTAL'] = txn_Q6['PRICE']*txn_Q6['STOCKS']
txn_Q7['TOTAL'] = txn_Q7['PRICE']*txn_Q7['STOCKS']
txn_Q8['TOTAL'] = txn_Q8['PRICE']*txn_Q8['STOCKS']
txn_Q9['TOTAL'] = txn_Q9['PRICE']*txn_Q9['STOCKS']



# loading stock_info
stock_Q1 = pd.read_csv('stock_info_2019Q1.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q2 = pd.read_csv('stock_info_2019Q2.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q3 = pd.read_csv('stock_info_2019Q3.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q4 = pd.read_csv('stock_info_2019Q4.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q5 = pd.read_csv('stock_info_2020Q1.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q6 = pd.read_csv('stock_info_2020Q2.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q7 = pd.read_csv('stock_info_2020Q3.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q8 = pd.read_csv('stock_info_2020Q4.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])
stock_Q9 = pd.read_csv('stock_info_2021Q1.csv',index_col=1,names=['DATE','STOCK_NO','OPEN','MAX','MIN','CLOSE','VOL','AMOUNT','CAPITAL','ALPHA','BETA_21D','BETA_65D','BETA_250D'])

# concat all stcok_info in a csv
stock_info = pd.concat([stock_Q1,stock_Q2,stock_Q3,stock_Q4,stock_Q5,stock_Q6,stock_Q7,stock_Q8,stock_Q9],axis=0)





# 計算每一位顧客在該季的交易次數、交易天數、交易金額 (input是單一季的txn檔案, output是三個向量)
def counting(txn):
    freq = []
    days = []
    price = []
    for i in trange(len(cust_info)):
        try:
            df = txn[txn.index==cust_info['cust'][i]]
            freq.append(len(df))
            days.append(len(df['DATE'].unique()))
            price.append(df['TOTAL'].sum())
        except KeyError:
            freq.append(0)
            days.append(0)
            price.append(0)
    return freq, days, price

# 自動化讓電腦自己把九個單季資料都算好並且併入到 cust_info.csv檔案裡面

txn_data = [txn_Q1,txn_Q2,txn_Q3,txn_Q4,txn_Q5,txn_Q6,txn_Q7,txn_Q8,txn_Q9]
txn_data_name = ['txn_Q1','txn_Q2','txn_Q3','txn_Q4','txn_Q5','txn_Q6','txn_Q7','txn_Q8','txn_Q9']
new_cust_info = cust_info
for (d,n) in zip(txn_data,txn_data_name):
    output = counting(d)
    for (i,j) in zip(range(3),['freq','day','price']):
        new_cust_info[n+'_'+j] = output[i]
    new_cust_info.to_csv('new_cust_info.csv')

##############################################################################
# 開始針對資料做一些視覺化統計
##############################################################################

# 查看每天的違約狀況
# 在10萬名顧客，568個交易日裡面，只有96筆違約交割(91個顧客)，分散在85個交易日內
# 圖片： 橫軸為交易日 ，縱軸為 違約交割的數量
breach = cust_info[cust_info['breach_ind']==1]
breach_count = []
for i  in trange(568):
    breach_count.append(len(breach[breach['breach_date']==i+1]))
plt.figure(figsize=(30,3))
plt.xlabel('date')
plt.plot(breach_count)


# 但是後來想到 10萬個客戶並不是每天都會交易，再者，可能有一些是幽靈客戶，根本都沒在下單的，應該把他們排除掉
# 把分母縮小到 ： 該日有下單記錄的客戶
# 橫軸為日期，縱軸為當天有下單的顧客之中違約交割的比例

# 有一個問題 ： txn的date最多只到 554, 可是違約紀錄的date 有到 560
cust_count = []
for i in trange(568):
    cust_count.append(len(txn[txn['DATE']==i+1].index.unique()))
breah_count = np.array(breach_count)
cust_count = np.array(cust_count)
bc_ratio = breach_count / cust_count
plt.figure(figsize=(30,3))
plt.xlabel('date')
plt.plot(bc_ratio)



# 年齡 age 的部分，對照整體用戶 與 違約交割用戶 ，發現年齡並不是影響違約交割的因素

# print(breach_age.size())
# print(all_age.size())
b = []
a = []
for i in range(1,7):
    b.append(len(breach[breach['age_level']==i]))
    a.append(len(cust_info[cust_info['age_level']==i]))
b = np.array(b)
b = b / b.sum()
a = np.array(a)
a = a / a.sum()
x = np.arange(6)
plt.title('age')
plt.bar(x,a,label='all',width=.3)
plt.bar(x+.3,b,label='breach',width=.3,color='orange')
plt.legend(loc = 'upper left')
plt.xticks(x+ .1  ,("1", "2", "3",'4','5','6'))



# 開戶日期 open 的部分，對照整體用戶 與 違約交割用戶 ，走勢大致上差不多，但可以發現2020年開戶的人真的違約交割比例比較嚴重
b = []
a = []
for i in range(1967,2022):
    b.append(len(breach[breach['open_acct_year']==i]))
    a.append(len(cust_info[cust_info['open_acct_year']==i]))
b = np.array(b)
b = b / b.sum()
a = np.array(a)
a = a / a.sum()
x = np.arange(1967,2022)
plt.title('Open Account')
plt.plot(x,a,label='all')
plt.plot(x,b,label='breach')
plt.legend(loc = 'upper left')
# print(b)



# 富果和玉山 source 的部分，對照整體用戶 與 違約交割用戶 ，看起來也不是影響違約交割的因素
b = []
a = []
li = ['A','B']
for i in li:
    b.append(len(breach[breach['source_code']==i]))
    a.append(len(cust_info[cust_info['source_code']==i]))
b = np.array(b)
b = b / b.sum()
a = np.array(a)
a = a / a.sum()
x = np.arange(2)
plt.title('Source')
plt.bar(x,a,label='all',width=.2)
plt.bar(x+.2,b,label='breach',width=.2,color='orange')
plt.xticks(x+ .3 / 2 ,('Fugle','E.SUN'))
plt.legend(loc = 'upper left')



# 交易經驗 txn_year 的部分，對照整體用戶 與 違約交割用戶 ，看得出來交易經驗未滿一年的人 比較容易違約交割
b = []
a = []
for i in range(5):
    b.append(len(breach[breach['investment_exp_code']==i]))
    a.append(len(cust_info[cust_info['investment_exp_code']==i]))
b = np.array(b)
b = b / b.sum()
a = np.array(a)
a = a / a.sum()
x = np.arange(5)
plt.title('TXN_Year')
plt.bar(x,a,label='all',width=0.4)
plt.bar(x+.2,b,label='breach',width=.3,color='orange')
plt.xticks(x+ .3 / 2 ,('0','1','2','3','4'))
plt.legend(loc = 'upper center')



# 交易的次數 buy_count 的部分，對照整體用戶 與 違約交割用戶 
breach[breach['buy_cnt']<200].groupby('buy_cnt').size().plot(label='breach',color = 'orange')
cust_info[cust_info['buy_cnt']<200].groupby('buy_cnt').size().plot(secondary_y=True,label='all')
# plt.legend(loc = 'best')



# 交易的次數 sell_count 的部分，對照整體用戶 與 違約交割用戶 
breach[breach['sell_cnt']<200].groupby('sell_cnt').size().plot()
cust_info[cust_info['sell_cnt']<200].groupby('sell_cnt').size().plot(secondary_y=True)



# 觀察違約交割的人 他們過去兩年的交易金額和交易頻率
new_cust_info['total_freq'] = new_cust_info['txn_Q1_freq']+new_cust_info['txn_Q2_freq']+new_cust_info['txn_Q3_freq']+new_cust_info['txn_Q4_freq']+new_cust_info['txn_Q5_freq']+new_cust_info['txn_Q6_freq']+new_cust_info['txn_Q7_freq']+new_cust_info['txn_Q8_freq']+new_cust_info['txn_Q9_freq']
new_cust_info['total_price'] = new_cust_info['txn_Q1_price']+new_cust_info['txn_Q2_price']+new_cust_info['txn_Q3_price']+new_cust_info['txn_Q4_price']+new_cust_info['txn_Q5_price']+new_cust_info['txn_Q6_price']+new_cust_info['txn_Q7_price']+new_cust_info['txn_Q8_price']+new_cust_info['txn_Q9_price']
new_cust_info['total_freq_rank'] = new_cust_info['total_freq'].rank(pct=True)
new_cust_info['total_price_rank'] = new_cust_info['total_price'].rank(pct=True)
breach = new_cust_info[new_cust_info['breach_ind']==1]



# 橫軸為交易金額的排名，縱軸為人數
tpr = []
tprc = [0]
pct = ['20%','40%','60%','80%','100%']
for i in range(5):
    c = len(breach[breach['total_price_rank'] > 1-((i+1)/5)] )
    tpr.append(c)
    tprc.append(c)
tpr = np.array(tpr)
tprc= np.array(tprc[:5])
tpr = tpr - tprc
plt.bar(pct,tpr,width=.4)
plt.title('total_price')




# 橫軸為交易次數的排名，縱軸為人數
tfr = []
tfrc = [0]
pct = ['20%','40%','60%','80%','100%']
for i in range(5):
    c = len(breach[breach['total_freq_rank'] > 1-((i+1)/5)] )
    tfr.append(c)
    tfrc.append(c)
tfr = np.array(tfr)
tfrc= np.array(tfrc[:5])
tfr = tfr - tfrc
plt.bar(pct,tfr,width=.4)
plt.title('total_freq')



#########################################################
# 建立 training data,  testing data
#########################################################

# 計算該客戶554天 每天的交割金額為多少 output是一個554維的向量  正數是支付，負數是拿回
# 計算該客戶554天 每天的交易次數為多少 output是一個554維的向量 
# 當天、前一天、前兩天的資料都納入參考
def breach_pay(cust):
    pay = np.zeros(554)
    times = np.zeros(554)
    pay_1 = np.zeros(554)
    pay_2 = np.zeros(554)
    times_1 = np.zeros(554)
    times_2 = np.zeros(554)
#     past = np.zeros(554)
    target = np.zeros(554)
    qwe = txn[txn.index==str(cust)]
    qw = breach[breach['cust']==str(cust)]
    for i in range(len(qwe)):
        try:
#         print(i)
            times[int(qwe.DATE[i]-1)] += 1
#         past[int(qwe.DATE[i]-1)] = int(qwe.DATE[i]-1 in qwe.DATE.tolist()) + int(qwe.DATE[i]-2 in qwe.DATE.tolist())
            if qwe.BS[i] == 'B':
                pay[int(qwe.DATE[i]-1)] += qwe.PRICE[i]*qwe.STOCKS[i]
            if qwe.BS[i] == 'S':
                pay[int(qwe.DATE[i]-1)] -= qwe.PRICE[i]*qwe.STOCKS[i]
            try :
                pay_1[int(qwe.DATE[i])] = pay[int(qwe.DATE[i]-1)]
#             pay_2[int(qwe.DATE[i]+1)] = pay[int(qwe.DATE[i]-1)]
                times_1[int(qwe.DATE[i])] = times[int(qwe.DATE[i]-1)]
#             times_2[int(qwe.DATE[i]+1)] = times[int(qwe.DATE[i]-1)]
            except IndexError :
                pass
            try :
#             pay_1[int(qwe.DATE[i])] = pay[int(qwe.DATE[i]-1)]
                pay_2[int(qwe.DATE[i]+1)] = pay[int(qwe.DATE[i]-1)]
#             times_1[int(qwe.DATE[i])] = times[int(qwe.DATE[i]-1)]
                times_2[int(qwe.DATE[i]+1)] = times[int(qwe.DATE[i]-1)]
            except IndexError :
                pass
        except ValueError:
            pass
        
    for i in range(len(qw)):
        try:
            target[int(qw.BREACH_DATE.tolist()[i]-1)] = 1
        except IndexError:
            pass
    if len(qwe) != 0:
        pay = (pay / np.abs(pay).max()) * 100
#     times /= times.max()
        pay_1 = (pay_1 / np.abs(pay_1).max()) * 100
#     times_1 /= times_1.max()
        pay_2 = (pay_2 / np.abs(pay_2).max()) * 100
#     times_2 /= times_2.max()
    
    return pay, times, pay_1, times_1, pay_2, times_2, target 

 

# 把這25位的資料切出來做成csv 並且切 training, testing
def makecsv(cust_list, train_size=400, test_size=154):
    train_pay = np.array([])
    train_times = np.array([])
    train_pay_1 = np.array([])
    train_times_1 = np.array([])
    train_pay_2 = np.array([])
    train_times_2 = np.array([])
    train_target = np.array([])
    
    test_pay = np.array([])
    test_times = np.array([])
    test_pay_1 = np.array([])
    test_times_1 = np.array([])
    test_pay_2 = np.array([])
    test_times_2 = np.array([])
    
    test_target = np.array([])
    
#     train_data = pd.DataFrame(columns=['pay','times','pay_1','times_1','pay_2','times_2','target'])
#     dev_data = pd.DataFrame(columns=['pay','times','pay_1','times_1','pay_2','times_2','target'])
#     test_data = pd.DataFrame(columns=['pay','times','pay_1','times_1','pay_2','times_2'])
#     test_answer = pd.DataFrame(columns=['target'])
    for i in trange(len(cust_list)):
#         print(i)
        data = breach_pay(cust_list.index[i])
        
        train_pay = np.append(train_pay,data[0][:train_size])
        train_times = np.append(train_times,data[1][:train_size])
        train_pay_1 = np.append(train_pay_1,data[2][:train_size])
        train_times_1 = np.append(train_times_1,data[3][:train_size])
        train_pay_2 = np.append(train_pay_2,data[4][:train_size])
        train_times_2 = np.append(train_times_2,data[5][:train_size])
        train_target = np.append(train_target,data[6][:train_size])
      
        test_pay = np.append(test_pay,data[0][train_size:train_size+test_size])
        test_times = np.append(test_times,data[1][train_size:train_size+test_size])
        test_pay_1 = np.append(test_pay_1,data[2][train_size:train_size+test_size])
        test_times_1 = np.append(test_times_1,data[3][train_size:train_size+test_size])
        test_pay_2 = np.append(test_pay_2,data[4][train_size:train_size+test_size])
        test_times_2 = np.append(test_times_2,data[5][train_size:train_size+test_size])
        test_target = np.append(test_target,data[6][train_size:train_size+test_size])
        
    train_data = {'pay':train_pay,
                 'times':train_times,
                 'pay_1':train_pay_1,
                 'times_1':train_times_1,
                 'pay_2':train_pay_2,
                 'times_2':train_times_2,
                 'target':train_target}
    train_data = pd.DataFrame(train_data)
    
    test_data = {'pay':test_pay,
                 'times':test_times,
                 'pay_1':test_pay_1,
                 'times_1':test_times_1,
                 'pay_2':test_pay_2,
                 'times_2':test_times_2}
    test_data = pd.DataFrame(test_data)
    
    test_answer = {'target':test_target}
    test_answer = pd.DataFrame(test_answer)
    
    
    
    train_data.to_csv('train_data.csv')
    test_data.to_csv('test_data.csv')
    test_answer.to_csv('test_answer.csv')
        