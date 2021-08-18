from kivy.uix.label import Label
from kivymd.app import MDApp
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from kivy.lang import Builder
import pickle
import numpy as np
from kivy.uix.widget import Widget
from yahoofinancials import YahooFinancials
import os
from kivy.uix.popup import Popup
popup = Popup(title='Error :(',
    content=Label(text='Hello world'),
    size_hint=(None, None), size=(300, 200))


Data=[[0 for x in range(3)] for y in range(7)]
StockPrice=[0 for x in range(7)]
def get_stock_data(stock):
    yahoo_financials = YahooFinancials(stock)
    data = yahoo_financials.get_historical_price_data(start_date='2020-01-01',
                                                      end_date=datetime.today().strftime('%Y-%m-%d'),
                                                      time_interval='daily')
    #get data from Yahoo Financials
    df = pd.DataFrame(data[stock]['prices']) #put Yahoo Financials Data to Dataframe
    y = df['close']
    df = df.drop('date', axis=1).set_index('formatted_date')
    x = pd.to_datetime(df.index)
    fig, ax = plt.subplots()    #instantiate subplots in order to allow the ability to plot multiple plots
    sc = plt.scatter(x, y, s=5, color='blue') #add scatter plot
    ax.set_xlabel('Date')   #set x label
    ax.set_ylabel('Stock Price (USD)') #set y label
    ax = plt.gca()  #get current axes from plot
    to_use = x[::19]    #skips date value of x in order to prevent xticks from overfilling
    labels=[i.strftime("%Y-%m-%d") for i in to_use] #changing format
    ax.set_xticks(labels) #set xticks to the formatted label
    price = np.array(y.round(3)).astype(str) #round the value of the y for the popup
    dates = np.array(df.index) #get index of dataframe
    #create annonate at point xy
    annot = ax.annotate("", xy=(0, 0), xycoords='data', xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc='white'),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    #used to update annenote when a scatter point is hovered over
    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format("\n".join("Price: "+price[n] for n in ind["ind"]),
                               "".join("\nDate: "+dates[n] for n in ind["ind"]))

        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(1.0)
    #triggers visibility when a point is hovered over
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    #connecting the motion notify event to the hover function
    fig.canvas.mpl_connect("motion_notify_event", hover)
    #rotate the xticks by 90 degress
    plt.xticks(rotation=90)
    #set legend to top left
    plt.legend(loc='upper left')
    #change plot title
    plt.title(stock+"'s Stock Data from 1st Jan 2020")

def add_predicted_point(date,price):

    plt.scatter(date,price, c='y', label='Predicted Result')
    plt.legend(loc='upper left')
    plt.title("Company's Stock Data and Predicted Result")

def get_today_stock(stock):
    if datetime.today().hour >21:
        start_date = (datetime.today()-timedelta(1)).strftime('%Y-%m-%d')
    else:
        start_date = datetime.today().strftime('%Y-%m-%d')
    yahoo_financials = YahooFinancials(stock)
    data = yahoo_financials.get_historical_price_data(start_date=(start_date),
                                                      end_date=datetime.today().strftime('%Y-%m-%d'),
                                                      time_interval='daily')

    df = pd.DataFrame(data[stock]['prices'])
    temp = df['close'].to_string()
    return temp[1:]

def show_graph():
    plt.show()

def get_pickle_model(company):
    root = os.path.dirname(os.path.abspath(__file__))
    filename = "models\\" + company + ".pickle"

    if os.path.getsize(filename) > 0:
        with open(os.path.join(root, filename), "rb") as f:
            unpickler = pickle.Unpickler(f)
            model = unpickler.load()
    return model
#C:\Users\jason\PycharmProjects\FYP App\models
def predict_stock(low,high,opens,volume,adjcls,company,stock):
    get_stock_data(stock)

    array = np.array([[high,low,opens,volume,adjcls]]).astype(float)

    model = get_pickle_model(company)

    result = model.predict_models(array)
    temp = result[0]
    predicted_result = str(round(temp[0],3))
    date= (datetime.today()+timedelta(1)).strftime('%Y-%m-%d')
    add_predicted_point(date,result)
    show_graph()
    return predicted_result
def get_trained_model(company):
    model = get_pickle_model(company)
    model.showRes()
def num_chk(var):
    try:
        float(var)
    except:
        return False
    return True

def Initialize():
    global Data
    count =0
    try:
        with open('userInfo.txt', 'r') as f:
            for line in f:
                line = line.strip()
                Data[count] = line.split('|',3)
                Data[count].pop()
                count += 1

    except:
        print("File not Found")

    StockPrice[0] = get_today_stock("TSLA")
    StockPrice[1] = get_today_stock("SONY")
    StockPrice[2] = get_today_stock("LPL")
    StockPrice[3] = get_today_stock("HYMTF")
    StockPrice[4] = get_today_stock("GOOG")
    StockPrice[5] = get_today_stock("DEA")
    StockPrice[6] = get_today_stock("NTDOY")



Builder.load_file('NavBar.kv')
Initialize()
class MyLayout(Widget):
    def show_trained(self,company):
        get_trained_model(company)
    def add_tesla_history(self):
        self.ids.stock_amt_tesla.text = str(Data[0][1])
        self.ids.stock_buy_prc_tesla.text = str(Data[0][2])
    def add_sony_history(self):
        self.ids.stock_amt_sony.text = str(Data[1][1])
        self.ids.stock_buy_prc_sony.text = str(Data[1][2])
    def add_lg_history(self):
        self.ids.stock_amt_lg.text = str(Data[2][1])
        self.ids.stock_buy_prc_lg.text = str(Data[2][2])
    def add_hyundai_history(self):
        self.ids.stock_amt_hyundai.text = str(Data[3][1])
        self.ids.stock_buy_prc_hyundai.text = str(Data[3][2])
    def add_google_history(self):
        self.ids.stock_amt_google.text = str(Data[4][1])
        self.ids.stock_buy_prc_google.text = str(Data[4][2])
    def add_dea_history(self):
        self.ids.stock_amt_dea.text = str(Data[5][1])
        self.ids.stock_buy_prc_dea.text = str(Data[5][2])
    def add_nintendo_history(self):
        self.ids.stock_amt_nintendo.text = str(Data[6][1])
        self.ids.stock_buy_prc_nintendo.text = str(Data[6][2])

    def update_home(self):

        self.ids.home_book_tesla.text = str(Data[0][0])
        self.ids.home_book_sony.text = str(Data[1][0])
        self.ids.home_book_lg.text = str(Data[2][0])
        self.ids.home_book_hyundai.text = str(Data[3][0])
        self.ids.home_book_google.text = str(Data[4][0])
        self.ids.home_book_dea.text = str(Data[5][0])
        self.ids.home_book_nintendo.text = str(Data[6][0])
        self.ids.home_prc_tesla.text = StockPrice[0]
        self.ids.home_prc_sony.text = StockPrice[1]
        self.ids.home_prc_lg.text = StockPrice[2]
        self.ids.home_prc_hyundai.text = StockPrice[3]
        self.ids.home_prc_google.text = StockPrice[4]
        self.ids.home_prc_dea.text = StockPrice[5]
        self.ids.home_prc_nintendo.text = StockPrice[6]
        self.ids.home_prof_tesla.text = str((float(Data[0][1]) * float(StockPrice[0])) - (float(Data[0][1]) * float(Data[0][2])))
        self.ids.home_prof_sony.text = str((float(Data[1][1]) * float(StockPrice[1])) - (float(Data[1][1]) * float(Data[1][2])))
        self.ids.home_prof_lg.text = str((float(Data[2][1]) * float(StockPrice[2])) - (float(Data[2][1]) * float(Data[2][2])))
        self.ids.home_prof_hyundai.text = str((float(Data[3][1]) * float(StockPrice[3])) - (float(Data[3][1]) * float(Data[3][2])))
        self.ids.home_prof_google.text = str((float(Data[4][1]) * float(StockPrice[4])) - (float(Data[4][1]) * float(Data[4][2])))
        self.ids.home_prof_dea.text = str((float(Data[5][1]) * float(StockPrice[5])) - (float(Data[5][1]) * float(Data[5][2])))
        self.ids.home_prof_nintendo.text = str((float(Data[6][1]) * float(StockPrice[6])) - (float(Data[6][1]) * float(Data[6][2])))

    def update_buy(self, company, amt, buy_prc):
        global Data
        if num_chk(amt) and num_chk(buy_prc):
            if Data[company-1][0]:
                temp = [Data[company-1][0],amt,buy_prc]
                Data[company-1] = temp
            else:
                Data[company-1][1] = amt
                Data[company-1][2] = buy_prc

            with open('userInfo.txt', 'w') as f:
                for data in Data:
                    for nuances in data:
                        f.write(str(nuances))
                        f.write('|')
                    f.write('\n')
            f.close()
            popup.title = 'Updated Successfully :)'
            popup.content = Label(text='New Values: \nAmount: '+Data[company-1][1]+'\nPurchase Price:'+Data[company-1][2])
            popup.open()

        else:
            popup.title = 'Error :('
            popup.content = Label(text='Please Enter Numerical Values')
            popup.open()
    def checkbox_click(self, instance, value, stock):
        global Data
        text = ""
        if (stock == "TSLA"):
            if value:
                text ="Yes"
            else:
                text ="No"
            self.ids.home_book_tesla.text = text
            Data[0][0] = text
        elif (stock == "SONY" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_sony.text = text
            Data[1][0] = text
        elif (stock == "DEA" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_dea.text = text
            Data[5][0] = text
        elif (stock == "LPL" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_lg.text = text
            Data[2][0] = text
        elif (stock == "HYMTF" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_hyundai.text = text
            Data[3][0] = text
        elif (stock == "GOOG" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_google.text = text
            Data[4][0] = text
        elif (stock == "NTDOY" ):
            if value:
                text = "Yes"
            else:
                text = "No"
            self.ids.home_book_nintendo.text = text
            Data[6][0] = text
        with open('userInfo.txt', 'w') as f:
            for data in Data:
                for nuances in data:
                    f.write(str(nuances))
                    f.write('|')
                f.write('\n')
        f.close()
    def show_stock_now(self, id):
        get_stock_data(id)
        show_graph()
    def button_predict_stock(self,low,high,opens,vol,adjcls,company,stock):
        if(num_chk(low) and num_chk(high) and num_chk(vol) and num_chk(adjcls) and num_chk(opens) ):
            price = predict_stock(low,high,opens,vol,adjcls,company,stock)
            text = "Predicted Result:  " + price + " USD"
            print(float(price))
            if (float(price) < 0.0):
                text ="Invalid Result, Please Try Again"

            if(stock == "TSLA"):
                self.ids.result_tesla.text=text
            elif (stock == "SONY"):
                self.ids.result_sony.text = text
            elif (stock == "DEA"):
                self.ids.result_dea.text = text
            elif (stock == "LPL"):
                self.ids.result_lg.text = text
            elif (stock == "HYMTF"):
                self.ids.result_hyundai.text = text
            elif (stock == "GOOG"):
                self.ids.result_google.text = text
            elif (stock == "NTDOY"):
                self.ids.result_nintendo.text = text
            popup.title="Predicted Results :)"
            popup.content = Label(text='Inputs:\nHigh: '+high+'\nLow: '+low+'\nOpen: '+opens+'\nVolume: '+vol+'\nAdjusted Close: '+adjcls+'\n Predicted Close Price: '+price)
            popup.open()
        else:
            popup.title = 'Error :('
            popup.content=Label(text='Please Enter Numerical Values')
            popup.open()
from kivy.clock import Clock
class App(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        return MyLayout()


    def on_start(self, **kwargs):
        Clock.schedule_once(self.initialize)
    def initialize(self, *args):
        self.root.ids.home_book_tesla.text = str(Data[0][0])
        self.root.ids.home_book_sony.text = str(Data[1][0])
        self.root.ids.home_book_lg.text = str(Data[2][0])
        self.root.ids.home_book_hyundai.text = str(Data[3][0])
        self.root.ids.home_book_google.text = str(Data[4][0])
        self.root.ids.home_book_dea.text = str(Data[5][0])
        self.root.ids.home_book_nintendo.text = str(Data[6][0])
        self.root.ids.home_prc_tesla.text = StockPrice[0]
        self.root.ids.home_prc_sony.text = StockPrice[1]
        self.root.ids.home_prc_lg.text = StockPrice[2]
        self.root.ids.home_prc_hyundai.text = StockPrice[3]
        self.root.ids.home_prc_google.text = StockPrice[4]
        self.root.ids.home_prc_dea.text = StockPrice[5]
        self.root.ids.home_prc_nintendo.text = StockPrice[6]
        
        self.root.ids.home_prof_tesla.text = str(
            (float(Data[0][1]) * float(StockPrice[0])) - (float(Data[0][1]) * float(Data[0][2])))
        self.root.ids.home_prof_sony.text = str(
            (float(Data[1][1]) * float(StockPrice[1])) - (float(Data[1][1]) * float(Data[1][2])))
        self.root.ids.home_prof_lg.text = str(
            (float(Data[2][1]) * float(StockPrice[2])) - (float(Data[2][1]) * float(Data[2][2])))
        self.root.ids.home_prof_hyundai.text = str(
            (float(Data[3][1]) * float(StockPrice[3])) - (float(Data[3][1]) * float(Data[3][2])))
        self.root.ids.home_prof_google.text = str(
            (float(Data[4][1]) * float(StockPrice[4])) - (float(Data[4][1]) * float(Data[4][2])))
        self.root.ids.home_prof_dea.text = str(
            (float(Data[5][1]) * float(StockPrice[5])) - (float(Data[5][1]) * float(Data[5][2])))
        self.root.ids.home_prof_nintendo.text = str(
            (float(Data[6][1]) * float(StockPrice[6])) - (float(Data[6][1]) * float(Data[6][2])))


App().run()
