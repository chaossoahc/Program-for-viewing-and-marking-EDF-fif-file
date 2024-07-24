import sys
import threading
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time
import PySimpleGUI as sg
from random import randint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# sg.theme('Default')
sg.theme('DarkBlack')
# sg.theme('DarkGrey4')

class App:
    """
    TODO: change slider resolution based on vid length
    TODO: make top menu actually do something :P
    TODO: изменить оборот ползунка в зависимости от длины видео TODO: заставить верхнее меню что-то делать :P
    """

    def __init__(self):
        # ------ App states ------ #
        self.play1 = False  # Is the video currently playing?Воспроизводится ли видео в данный момент?
        self.play2 = False  # Is the video currently playing?Воспроизводится ли видео в данный момент?
        self.delay1 = 0.0166  # Задержка между кадрами
        self.delay2 = 0.0166  # Задержка между кадрами
        self.frame1 = 1  # Current frame Текущий кадр
        self.frame2 = 1  # Current frame Текущий кадр
        self.frames1 = None  # Number of frames Количество кадров
        self.frames2 = None  # Number of frames Количество кадров
        self.duration1 = None
        self.data_flag = False
        self.video1_flag = False
        self.video2_flag = False
        self.flag_metki=False
        self.used_enables_channels=False
        # root = tk.Tk()
        # ------ Other vars ------ #

        # sizewindow = [root.winfo_screenwidth(),root.winfo_screenheight()]
        # root.destroy()
        # print(sizewindow)
        sizewindow = [1920,1080]
        #sizewindow = [2560,1440]   
        # sizewindow = [2560,1440]
        self.sizewindow = sizewindow
        if round(self.sizewindow[0]/self.sizewindow[1])==1.3:
            self.thickness = 3
            self.thickness_whide = 4
            koef_lenght = self.sizewindow[0]/self.thickness_whide
            koef_width = self.sizewindow[1]/self.thickness

        else:
            koef = (self.sizewindow[0] * self.sizewindow[1])/(1920*1080)
            self.thickness = 10*(koef-koef*0.1)
            self.thickness_whide = 18*(koef-koef*0.1)
            koef_lenght = self.sizewindow[0]/self.thickness_whide
            koef_width = self.sizewindow[1]/self.thickness

        self.vid1 = None
        self.vid2 = None
        self.photo1 = None
        self.photo2 = None
        self.next1 = "1"
        self.next2 = "1"
      
        def size_ekran(sizewindow):
            if sizewindow == [2560,1440]:
                self.dpi = 70
                self.slider_eeg = 5
                self.length_slider_EEG = 100 # 100 = 1920*x x = 100/1920
                self.length_slider_BIG = 150
                self.length_slider_video = 50
                self.width_slider_video = 10
                self.width_slider_EEG = 10
                self.vid_width1 = 400
                self.vid_width2 = 400

            else:

                self.dpi = 70/sizewindow[0]*sizewindow[0]
                self.slider_eeg = 5
                self.length_slider_EEG = 100/sizewindow[0]*sizewindow[0]
                self.length_slider_BIG = 150/sizewindow[0]*sizewindow[0]
                self.length_slider_video = 50/sizewindow[0]*sizewindow[0]
                self.width_slider_video = 10/sizewindow[0]*sizewindow[0]
                self.width_slider_EEG = 10/sizewindow[0]*sizewindow[0]
                self.vid_width1 = int(400/sizewindow[0]*sizewindow[0])
                self.vid_width2 = int(400/sizewindow[0]*sizewindow[0])
        size_ekran(sizewindow)

        font = ("Arial", 8)
        sg.set_options(font=font)
        layout_lef = [[sg.Text('Текущй отсчёт',grab=True,font=font)],
                      [sg.Text('Ширина окна',grab=True)],
                      [sg.Text('Предела массштаба окна',grab=True)],
                      [sg.Text('Промотка по Каналам',grab=True)],
                      [sg.Text('Расстояние между каналами',grab=True)]]
        
        layout_rig = [[sg.Slider(range=(0, 100), size=(self.length_slider_EEG, self.width_slider_EEG), resolution=1, orientation='h', key='-SLIDER-',disable_number_display=True, enable_events=True), sg.T("0", key="counter_EEG", size=(10, 1))],
                     [sg.Slider(range=(1, 60), default_value=5, size=(self.length_slider_EEG, self.width_slider_EEG), orientation='h', key='-SLIDER-DATAPOINTS-',disable_number_display=True, enable_events=True), sg.T("0", key="window_EEG", size=(10, 1))],
                       
                       [sg.Slider(range=(0, 10000000, 1), default_value=1, size=(self.length_slider_EEG, self.width_slider_EEG), orientation='h', key='-SLIDER-scaleDATAPOINTS[-]-',disable_number_display=True,enable_events=True)],
                       [sg.Slider(range=(1, 100000), default_value=1150, size=(self.length_slider_EEG, self.width_slider_EEG),orientation='h', key='-SLIDER-scalePOINTS-',disable_number_display=True, enable_events=True)]]
        
        layout_l = [[sg.Canvas(size=(200, 200), key="canvas3", background_color='yellow')],
                    [sg.Col(layout_lef), sg.Col(layout_rig)]]

        layout_r =[[sg.Input(key="_FILEPATH_1"), sg.Button("Видео 1")],
                    [sg.Canvas(size=(200, 200), key="canvas1", background_color='green')],
                    [sg.Slider(size=(45, 10), range=(0, 100), resolution=1, key="1 slider", orientation="h",disable_number_display=True, enable_events=True), sg.T("0", key="1 counter", size=(10, 1))],
                    [sg.Button('1 Back 1 second'), sg.Button('1 Back frame'), sg.Button('1 Next frame'), sg.Button('1 Next 1 second')],
                    [sg.Input(key="_FILEPATH_2"), sg.Button("Видео 2")],
                    [sg.Canvas(size=(200, 200), key="canvas2", background_color='green')],
                    [sg.Slider(size=(45, 10), range=(0, 100), resolution=1, key="2 slider", orientation="h",disable_number_display=True, enable_events=True), sg.T("0", key="2 counter", size=(10, 1))],
                    [sg.Button('2 Back 1 second'),sg.Button('2 Back frame'),sg.Button('2 Next frame'), sg.Button('2 Next 1 second')]]
                    # [sg.Input(key="_FILEPATH_3"), sg.Button("Аудио")],
                    # [sg.Canvas(size=(200, 200), key="canvas4", background_color='red')]]


        layout =[[sg.Button("Открыть ЭЭГ"), sg.Button("перейти к фрагменту", disabled=True), sg.Button('выбрать каналы'), sg.Button('загрузить метки'), sg.Button("Открыть ANN", disabled=True), sg.Button('добавить метку'), sg.Button('сохранить сигнал', disabled=True), sg.Button('построить спектр', disabled=True), sg.Button('сохранить метки', disabled=True), sg.Button('Exit')],
                 [sg.Col(layout_l), sg.Col(layout_r)],
                 [sg.Slider(range=(1, 100000, 1), default_value=1, size=(self.length_slider_BIG, self.width_slider_EEG), orientation='h', key='-SLIDER-BIG-',disable_number_display=True,enable_events=True),sg.Button('back 1 second'),  sg.Button('next 1 second')]]
        self.window = sg.Window('The PySimpleGUI Element List', layout, finalize=True, resizable=True)
        # self.window = sg.Window('Window Title', layout).Finalize()
        # set return_keyboard_events=True to make hotkeys for video playback установите return_keyboard_events = True, чтобы сделать горячие клавиши для воспроизведения видео
        # Get the tkinter canvas for displaying the video Получите холст tkinter для отображения видео
        canvas1 = self.window.Element("canvas1")
        canvas2 = self.window.Element("canvas2")
        canvas3 = self.window.Element("canvas3")
        self.canvas1 = canvas1.TKCanvas
        self.canvas2 = canvas2.TKCanvas
        self.canvas3 = canvas3.TKCanvas
        # Start video display thread Начать ветку показа видео
        self.load_video1()
        self.load_video2()

        while True:  # Main event Loop
            event, values = self.window.Read()

            if event is None or event == 'Exit':
                """Handle exit"""
                break

            if event == "-SLIDER-":
                self.SLIDER(int(values["-SLIDER-"]))

            if event == "-SLIDER-BIG-":
                self.SLIDER_BIG(int(values["-SLIDER-BIG-"]))

                if self.data_flag:
                    self.SLIDER(int(values["-SLIDER-BIG-"]*self.sampling_frequency))
    
                    self.window.Element("-SLIDER-").Update(value=int(values["-SLIDER-BIG-"]*self.sampling_frequency))

                if self.video1_flag:
                    self.set_frame1(int(values["-SLIDER-BIG-"])*self.fps1)
                if self.video2_flag:
                    self.set_frame2(int(values["-SLIDER-BIG-"])*self.fps2)
            if event == "back 1 second":
                self.SLIDER(int(values["-SLIDER-BIG-"]*self.sampling_frequency))
                self.window.Element("-SLIDER-BIG-").Update(value=int(values["-SLIDER-BIG-"]) - 1)
                self.window.Element("-SLIDER-").Update(value=int(values["-SLIDER-BIG-"]) - 1)
                if self.video1_flag:
                    self.set_frame1(self.frame1 - 1*self.fps1)
                if self.video2_flag:
                    self.set_frame2(self.frame2 - 1*self.fps2)
                
                
            if event == "next 1 second":

                self.SLIDER(int(values["-SLIDER-BIG-"]*self.sampling_frequency))
                self.window.Element("-SLIDER-BIG-").Update(value=int(values["-SLIDER-BIG-"]) + 1)
                self.window.Element("-SLIDER-").Update(value=int(values["-SLIDER-BIG-"]) + 1)
                if self.video2_flag:
                    self.set_frame2(self.frame2 + 1*self.fps2)
                if self.video1_flag:
                    self.set_frame1(self.frame1 + 1*self.fps1)

            if event == "Видео 1":
                """Кнопка файлового браузера"""
                # открытие файлового диалога
                video_path1 = None
                try:
                    video_path1 = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("видео не выбрано! Пользователь нажал отмену")
                if video_path1:
                    self.video1_flag = True
                    # Инициализировать видео
                    self.vid1 = MyVideoCapture1(video_path1)
                    # Рассчитать новые размеры видео
                    self.fps1=self.vid1.fps1
                    self.vid_width1 = self.vid_width1
                    self.vid_height1 = int(self.vid_width1 * self.vid1.height1 / self.vid1.width1)
                    self.frames1 = int(self.vid1.frames1)
                    # Обновите ползунок, чтобы он соответствовал количеству кадров.
                    self.window.Element("1 slider").Update(range=(0, int(self.frames1)), value=0)
                    # Обновить правую часть стойки
                    self.window.Element("1 counter").Update("0/%i" % self.frames1)
                    # изменить размер холста примерно на размер видео
                    self.canvas1.config(width=self.vid_width1, height=self.vid_height1)
                    # Сбросить счетчик кадров
                    self.frame1 = 0
                    self.delay1 = round(1 / self.vid1.fps1, 5)
                    # Обновите текстовое поле пути к видео.
                    self.window.Element("_FILEPATH_1").Update(video_path1)

            if event == "Видео 2":
                
                """Кнопка файлового браузера"""
                # открытие файлового диалога
                video_path2 = None
                try:
                    video_path2 = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("видео не выбрано! Пользователь нажал отмену")

                if video_path2:
                    self.video2_flag = True
                    # иницилизация видео
                    self.vid2 = MyVideoCapture2(video_path2)
                    # Рассчитать новые размеры видео
                    self.vid_width2 = self.vid_width2
                    self.vid_height2 = int(self.vid_width2 * self.vid2.height2 / self.vid2.width2)
                    self.frames2 = int(self.vid2.frames2)
                    self.fps2=self.vid2.fps2
                    # Обновите ползунок, чтобы он соответствовал количеству кадров.
                    self.window.Element("2 slider").Update(range=(0, int(self.frames2)), value=0)
                    # Обновить правую часть стойки
                    self.window.Element("2 counter").Update("0/%i" % self.frames2)
                    # изменить размер холста примерно на размер видео
                    self.canvas2.config(width=self.vid_width2, height=self.vid_height2)
                    # Сбросить счетчик кадров
                    self.frame2 = 0
                    self.delay2 = round(1 / self.vid2.fps2, 5)

                    self.window.Element("_FILEPATH_2").Update(video_path2)
            
            if event == "Открыть ЭЭГ":
                
                """Кнопка файлового браузера"""
                # открытие файлового диалога
                data_path = None
                try:
                    data_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("EDF не выбран! Пользователь нажал отмену")
                
                if data_path:
                    self.data_EEG = MyDataEEG(data_path)
                    self.all_ch_names= self.data_EEG.all_ch_names
                    self.all_ch_names_no_reverse= self.data_EEG.all_ch_names
                    self.all_ch_names.reverse()
                    channel_data = self.data_EEG.channel_data
                    self.EEG_Fp1_ = channel_data[:, :]
                    self.data_shape = self.data_EEG.data_shape

                    if self.data_EEG.disabled == self.data_EEG.disabled_EEG:
                        self.data_metki_otschet = self.data_EEG.data_metki_otschet
                        self.data_metki_label =self.data_EEG.data_metki_label
                        self.flag_metki = True

                    self.window['Открыть ANN'].update(disabled=self.data_EEG.disabled)
                    self.data_flag = True
                    self.window['Открыть ЭЭГ'].update(disabled=self.data_EEG.disabled_EEG)
                    # self.window.Element("-SLIDER-scaleDATAPOINTS[+]-").Update(value=self.data_EEG.data_value)
                    self.window['перейти к фрагменту'].update(disabled=False)
                    self.window['сохранить сигнал'].update(disabled=False)
                    self.window['построить спектр'].update(disabled=False)
                    self.window['сохранить метки'].update(disabled=False)
                    self.sampling_frequency = self.data_EEG.sampling_frequency
                    self.second_data = self.data_shape/self.sampling_frequency
                    self.window.Element("-SLIDER-BIG-").Update(range=(0, int(int(self.second_data))), value=0)
                    self.window.Element("-SLIDER-").Update(range=(0, int(self.data_shape)), value=0)
                    canvas_EEG_Fp1_ = self.canvas3
                    self.x_vector = np.arange(self.data_shape)
                    fig_EEG_Fp1_ = Figure(figsize=(self.thickness_whide, self.thickness), dpi=self.dpi)
                    self.ax_EEG_Fp1_ = fig_EEG_Fp1_.add_subplot()
                    fig_EEG_Fp1_.subplots_adjust(top=1,right=1,left=0.05, bottom=0.03)
                    self.ax_EEG_Fp1_.grid()
                    self.fig_agg_EEG_Fp1_ = self.draw_figure(canvas_EEG_Fp1_, fig_EEG_Fp1_)

            if event == "перейти к фрагменту":
                
                layout_perehod = [[sg.Text('Введите время для перехода')],
                            [sg.Text('час', size=(10, 1)), sg.InputText(''),],
                            [sg.Text('минута', size=(10, 1)), sg.InputText('')],
                            [sg.Text('секунда', size=(10, 1)), sg.InputText('')],
                            [sg.Submit("подтвердить"), sg.Cancel("отмена")]]

                window2_perehod = sg.Window('Rename Files or Folders', layout_perehod)
                event2, values2__perehod = window2_perehod.read()
                window2_perehod.close()
                print(values2__perehod)
                chas, minuta, sekunda = (values2__perehod[0]), (values2__perehod[1]), (values2__perehod[2])

                if chas != '':
                    chas = int(chas)
                else:
                    chas = 0
                if minuta != '':
                    minuta = int(minuta)
                else:
                    minuta = 0
                if sekunda != '':
                    sekunda = int(sekunda)
                else:
                    sekunda = 0
                
                
                self.SLIDER(int((chas*3600 + minuta*60 + sekunda)*self.sampling_frequency))
                self.window.Element("-SLIDER-").Update(value=int((chas*3600 + minuta*60 + sekunda)*self.sampling_frequency))


            if event == "сохранить сигнал":
                if self.data_EEG.disabled_EEG:
                    layout2 = [[sg.Text('без папки и имени файла не сохранит')],
                                [sg.Text('выберите папку', size=(15, 1)), sg.InputText(), sg.FolderBrowse()],
                                [sg.Text('введите имя файла', size=(15, 1)), sg.InputText()],
                                [sg.Submit(), sg.Cancel()]]

                    window2 = sg.Window('Rename Files or Folders', layout2)
                    event2, values2 = window2.read()
                    window2.close()
                    folder_path, file_path = values2[0], values2[1]       # get the data from the values dictionary

                    if folder_path is None or file_path is None:
                        print("нет имени файла и папки")
                    elif folder_path=='':
                        print("нет имени папки")
                    elif file_path =='':
                        print("нет имени файла")
                    else:
                        print(folder_path+'/'+file_path)
                
                        dick = {}
                        dick['otchet'] = self.otchet_save_to_txt
                        for i in range(len(self.all_ch_names)):
                            dick[self.all_ch_names[len(self.all_ch_names)-i-1]] = self.data_save_to_txt[i]

                        df = pd.DataFrame(dick)
                        df.to_csv(folder_path+'/'+file_path+'.csv') 

            if event == 'построить спектр':
                def batton_generation(all_ch_names):
                    layout_name = []
                    for name in all_ch_names:
                        layout_name.append([sg.Button(name)])
                    return layout_name
                layout_name2 = batton_generation(self.all_ch_names)
                window2 = sg.Window('Rename Files or Folders', layout_name2)
                event2, values2 = window2.read()

                def event_spektr(event2, all_ch_names, data_save_to_txt):
                    def c_sf_spm(arr):  # суммарный спектр
                        SVdft = fft(arr);
                        N_coef = int(SVdft.shape[0] / 2)
                        N = len(arr)
                        SVdft2 = SVdft[:N_coef + 1]  # %FFT - расчет КФ (комплексные)
                        psdSV = 2 * ((1 / (self.sampling_frequency * N)) * abs(SVdft2) ** 2)
                        return psdSV
                    index = all_ch_names.index(event2)
                    spektr = np.log(c_sf_spm(data_save_to_txt[len(all_ch_names)-index-1]))
                    plt.title(event2)
                    plt.xlabel('частота')
                    plt.ylabel('мощность (log)')
                    plt.plot(np.arange(0, int(self.sampling_frequency/2),int(self.sampling_frequency/2)/len(spektr[:-1])),spektr[:-1])
                    plt.show()
                    
                event_spektr(event2, self.all_ch_names, self.data_save_to_txt)
                self.data_save_to_txt 
                window2.close()
            
            if event == "сохранить метки":
                if self.data_EEG.disabled_EEG:
                    layout2 = [[sg.Text('без папки и имени файла не сохранит')],
                              [sg.Text('выберите папку', size=(15, 1)), sg.InputText(), sg.FolderBrowse()],
                              [sg.Text('введите имя файла', size=(15, 1)), sg.InputText()],
                              [sg.Submit("сохранить"), sg.Cancel("отмена")]]

                    window2 = sg.Window('Rename Files or Folders', layout2)
                    event2, values2 = window2.read()
                    window2.close()
                    folder_path_metki, file_path_metki = values2[0], values2[1]       # get the data from the values dictionary

                    if folder_path_metki is None or file_path_metki is None:
                        print("нет имени файла и папки")
                    elif folder_path_metki=='':
                        print("нет имени папки")
                    elif file_path_metki =='':
                        print("нет имени файла")
                    else:
                        pass

                    self.dict_metki = {}         
                    if self.data_flag:
                        self.dict_metki["data_otschet"] = self.data_metki_otschet
                        self.dict_metki["label"] = self.data_metki_label
                    if self.video1_flag:
                        self.dict_metki["frame_video_1"] = (self.data_metki_otschet/self.sampling_frequency*self.fps1).astype(int)
                    if self.video2_flag:
                        self.dict_metki["frame_video_2"] = (self.data_metki_otschet/self.sampling_frequency*self.fps2).astype(int)

                    df3 = pd.DataFrame(self.dict_metki)
                    df3.to_csv(folder_path_metki+'/'+file_path_metki+'.csv')

            if event == 'загрузить метки':
                load_metki = None
                try:
                    load_metki = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("метки не выбраны! Пользователь нажал отмену")
                if load_metki:
                    metki_load  = pd.read_csv(load_metki, delimiter = ',', encoding='cp1251')
                    self.data_metki_otschet = metki_load["data_otschet"].to_numpy()
                    self.data_metki_label = metki_load["label"].values.tolist()


            if event == "Открыть ANN":
                ann_path = None
                try:
                    ann_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("ANN фаил не выбран! Пользователь нажал отмену")
                
                if ann_path:
                    txt_ann_file = pd.read_csv(ann_path, delimiter = ',', header=None, encoding='cp1251',skiprows=1)
                    self.data_metki_otschet = txt_ann_file.iloc[:,0]
                    self.data_metki_label = txt_ann_file.iloc[:,-1]
                    self.flag_metki = True


            if event == '1 Next frame':
                self.set_frame1(self.frame1 + 1)
            if event == '1 Next 1 second':
                self.set_frame1(self.frame1 + 1*self.fps1)
            if event == '1 Back frame':
                self.set_frame1(self.frame1 - 1)
            if event == '1 Back 1 second':
                self.set_frame1(self.frame1 - 1*self.fps1)
            if event == "1 slider":
                self.set_frame1(int(values["1 slider"]))


            if event == "-SLIDER-DATAPOINTS-":
                self.SLIDER_DATAPOINTS(int(values["-SLIDER-DATAPOINTS-"]))
            if event == "-SLIDER-scaleDATAPOINTS[-]-":
                self.SLIDER_scaleDATAPOINTS(int(values["-SLIDER-scaleDATAPOINTS[-]-"]))
            if event == "-SLIDER-scalePOINTS-":
                self.SLIDER_scalePOINTS(int(values["-SLIDER-scalePOINTS-"]))

            if event == "выбрать каналы":
                def batton_generation(all_ch_names):
                    layout_name = []
                    for name in all_ch_names:
                        layout_name.append([sg.Checkbox(name, key=name, default=True)])
                    layout_name.append([sg.Submit(), sg.Cancel()])
                    return layout_name
                layout_name_checkbox = batton_generation(self.all_ch_names)
                window_checkbox = sg.Window('Rename Files or Folders', layout_name_checkbox)
                event_checkbox, values_checkbox = window_checkbox.read()
                window_checkbox.close()
                
                self.channel_list_enables = list(values_checkbox.keys())
                self.enables_chanels = []
                self.use_new_channel = []
                for enables_chan in self.channel_list_enables:
                    index = self.channel_list_enables.index(enables_chan)
                    if values_checkbox[enables_chan]:
                        
                        self.use_new_channel.append(enables_chan)
                        self.enables_chanels.append(index)
                self.used_enables_channels = True
                # print(self.enables_chanels, "self.enables_chanels")



            if event == 'добавить метку':
                layout_metki = [[sg.Text('введите имя метки', size=(15, 1)), sg.InputText()],
                                [sg.Submit(), sg.Cancel()]]

                window_metki = sg.Window('Rename Files or Folders', layout_metki)
                event_metki, values_metki = window_metki.read()
                window_metki.close()
                imya_metki = values_metki[0] 

                self.data_metki_otschet = np.append(self.data_metki_otschet, self.otchet_save_to_txt[0])
                self.data_metki_label.append(imya_metki)

            if event == '2 Next frame':
                self.set_frame2(self.frame2 + 1)
            if event == '2 Next 1 second':
                self.set_frame2(self.frame2 + 1*self.fps2)
            if event == '2 Back frame':
                self.set_frame2(self.frame2 - 1)
            if event == '2 Back 1 second':
                self.set_frame2(self.frame2 - 1*self.fps2)
            if event == "2 slider":
                self.set_frame2(int(values["2 slider"]))

            
            
             
        # Exiting
        print("Завершение работы")
        self.window.Close()
        sys.exit()

    def SLIDER(self, frame10):
        self.slider_eeg = frame10
        self.eeg_slider_update()

    def SLIDER_DATAPOINTS(self, frame20):
        self.data_points = frame20
        self.eeg_slider_update()

    def SLIDER_scaleDATAPOINTS(self, frame30):
        self.scale_data_points = frame30
        self.eeg_slider_update()

    def SLIDER_scalePOINTS(self, frame40):
        self.scale_points = frame40
        self.eeg_slider_update()

    def SLIDER_BIG(self, frame100):
        self.scale_data_points = frame100
        self.eeg_slider_update()

    def otschet_v_time(self, slider_eeg, sampling_frequency):
        otschet_duration = int(slider_eeg)%sampling_frequency
        secund_duration = int(slider_eeg)//sampling_frequency%60
        minut_duration = int(slider_eeg)//sampling_frequency//60%60
        hour_duration =  int(slider_eeg)//sampling_frequency//60//60%60

        if hour_duration==0 and minut_duration == 0 and secund_duration==0:
            self.time_iteration = str(int(otschet_duration))

        elif hour_duration==0 and minut_duration == 0:
            self.time_iteration = str(int(secund_duration))+':'+str(int(otschet_duration))

        elif hour_duration==0:
            self.time_iteration = str(int(minut_duration))+':'+ str(int(secund_duration))+':'+str(int(otschet_duration))

        else:
            self.time_iteration = str(int(hour_duration))+':'+ str(int(minut_duration))+':'+ str(int(secund_duration))+':'+str(int(otschet_duration))
        return self.time_iteration
   
    def eeg_slider_update(self):
        self.event, self.values = self.window.read(timeout=1)
        self.ax_EEG_Fp1_.cla()  # clear the subplot
        self.ax_EEG_Fp1_.grid()  # draw the grid

        for label in (self.ax_EEG_Fp1_.get_yticklabels()):
            label.set_fontsize(10)
        
        duration_slider_EEG = int(self.slider_eeg//self.sampling_frequency*self.sampling_frequency)
        time_iteration = self.otschet_v_time(duration_slider_EEG, self.sampling_frequency)
        self.window.Element("counter_EEG").Update(time_iteration)
        # scale_data_points_up = int(self.values['-SLIDER-scaleDATAPOINTS[+]-'])
        scale_data_points_up = 2300*len(self.all_ch_names)
        
        scale_data_points_down = int(self.values['-SLIDER-scaleDATAPOINTS[-]-'])
        scale_points = int(self.values['-SLIDER-scalePOINTS-'])
        self.data_points = int(self.values['-SLIDER-DATAPOINTS-']*self.sampling_frequency)  # draw this many data points (on next line)
        time_window = self.otschet_v_time(self.data_points, self.sampling_frequency)
        self.window.Element("window_EEG").Update(time_window)

        if duration_slider_EEG > self.data_shape-self.data_points:
            duration_slider_EEG = self.data_shape-self.data_points
        
        # color_line = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black'] # цвета линий
        ycoeff_list = []
        #self.channel_list_enables
        for chan in range(self.EEG_Fp1_.shape[0]):
            y_coeff = chan*scale_points*0.001
            ycoeff_list.append(y_coeff)

        ycoeff_np = np.array(ycoeff_list)

        if self.used_enables_channels:
            self.ax_EEG_Fp1_.set_yticks(ycoeff_np[:len(self.use_new_channel)])
            self.ax_EEG_Fp1_.set_yticklabels(self.use_new_channel)
            scale_data_points_up = 2300*len(self.use_new_channel)
        
        else:
            self.ax_EEG_Fp1_.set_yticks(ycoeff_np)
            self.ax_EEG_Fp1_.set_yticklabels(self.all_ch_names)

        
        

        if self.flag_metki==True:
            
                
            metki_v_okne = self.data_metki_otschet[(self.data_metki_otschet > duration_slider_EEG ) & (self.data_metki_otschet < (duration_slider_EEG+self.data_points ))]
            

            if len(metki_v_okne)>= 1:
                nachalo_okna = np.where(self.data_metki_otschet== metki_v_okne[0])[0][0]
                konetc_okna = np.where(self.data_metki_otschet== metki_v_okne[-1])[0][0]
                # print(type(self.data_metki_label))
                # print(konetc_okna, 'konetc_okna')

                self.ax_EEG_Fp1_.set_xticks(self.data_metki_otschet[nachalo_okna:konetc_okna+1])
                self.ax_EEG_Fp1_.set_xticklabels(self.data_metki_label[nachalo_okna:konetc_okna+1])


        # отображение
        cost_y_correct = 0
        for chan in range(self.EEG_Fp1_.shape[0]):
            try:
                
                if self.enables_chanels.count(chan)>0:
                    self.ax_EEG_Fp1_.set_ylim([-0.0015+0.00001*scale_data_points_down, 0.0005*scale_data_points_up+0.00001*scale_data_points_down])            
                    data_n = self.normal_set_get(self.EEG_Fp1_[self.EEG_Fp1_.shape[0]-1-chan][duration_slider_EEG:duration_slider_EEG + self.data_points])
                    self.ax_EEG_Fp1_.plot(self.x_vector[duration_slider_EEG:duration_slider_EEG+self.data_points], data_n+ycoeff_np[cost_y_correct], color='black')
                    cost_y_correct = cost_y_correct + 1
                else:
                    continue
            except:
                self.ax_EEG_Fp1_.set_ylim([-0.0015+0.00001*scale_data_points_down, 0.0005*scale_data_points_up+0.00001*scale_data_points_down])            
                data_n = self.normal_set_get(self.EEG_Fp1_[self.EEG_Fp1_.shape[0]-1-chan][duration_slider_EEG:duration_slider_EEG + self.data_points])
                self.ax_EEG_Fp1_.plot(self.x_vector[duration_slider_EEG:duration_slider_EEG+self.data_points], data_n+ycoeff_np[chan], color='black')
                                   
        self.fig_agg_EEG_Fp1_.draw()
        self.data_save_to_txt = self.EEG_Fp1_[:,duration_slider_EEG:duration_slider_EEG + self.data_points]
        self.otchet_save_to_txt = self.x_vector[duration_slider_EEG:duration_slider_EEG + self.data_points]

    def normal_set_get(self, data):
        if np.isnan(data).all():
            data = data
        else:
            scaler_get = MinMaxScaler()
            data = scaler_get.fit_transform(data.reshape(-1, 1))
            data = data[:, 0]
        return data
    
    def load_video1(self):
        """Start video display in a new thread Начать показ видео в новом потоке"""
        thread = threading.Thread(target=self.update1, args=())
        thread.daemon = 1
        thread.start()
    def load_video2(self):
        """Start video display in a new thread Начать показ видео в новом потоке"""
        thread = threading.Thread(target=self.update2, args=())
        thread.daemon = 2
        thread.start()


    def update1(self):
        """Update the canvas element with the next video frame recursively"""
        start_time = time.time()
        if self.vid1:
            if self.play1:
                # Get a frame from the video source only if the video is supposed to play
                ret1, frame1 = self.vid1.get_frame1()
                if ret1:
                    self.photo1 = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame1).resize((self.vid_width1, self.vid_height1), Image.NEAREST))
                    self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)
                    self.frame1 += 1
                    self.update_counter1(self.frame1)

        # Метод tkinter .after позволяет нам выполнить рекурсию после задержки, не достигая предела рекурсии. Нам нужно подождать
        # между каждым кадром для достижения нужного количества кадров в секунду, но также подсчитайте время, необходимое для создания предыдущего кадра.
        self.canvas1.after(abs(int((self.delay1 - (time.time() - start_time)) * 1000)), self.update1)

    def update2(self):
        
        """Update the canvas element with the next video frame recursively"""
        start_time = time.time()
        if self.vid2:
            if self.play2:
                # Get a frame from the video source only if the video is supposed to play
                ret2, frame2 = self.vid2.get_frame2()
                if ret2:
                    self.photo2 = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame2).resize((self.vid_width2, self.vid_height2), Image.NEAREST))
                    self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)
                    self.frame2 += 1
                    self.update_counter2(self.frame2)
        self.canvas2.after(abs(int((self.delay2 - (time.time() - start_time)) * 1000)), self.update2)


    def set_frame1(self, frame_no1):
        """Jump to a specific frame"""
        if self.vid1:
            # Get a frame from the video source only if the video is supposed to play
            ret1, frame1 = self.vid1.goto_frame1(frame_no1)
            self.frame1 = frame_no1
            
            self.update_counter1(self.frame1)

            if ret1:
                self.photo1 = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame1).resize((self.vid_width1, self.vid_height1), Image.NEAREST))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)

    def set_frame2(self, frame_no2):
        """Jump to a specific frame"""
        if self.vid2:
            # Get a frame from the video source only if the video is supposed to play
            ret2, frame2 = self.vid2.goto_frame2(frame_no2)
            self.frame2 = frame_no2
            self.update_counter2(self.frame2)

            if ret2:
                self.photo2 = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame2).resize((self.vid_width2, self.vid_height2), Image.NEAREST))
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)

    def update_counter1(self, frame1):
        """Helper function for updating slider and frame counter elements"""
        self.window.Element("1 slider").Update(value=frame1)
        time_iteration = self.otschet_v_time(frame1, self.fps1)
        self.window.Element("1 counter").Update("{}/{}".format(time_iteration, self.otschet_v_time(self.frames1, self.fps1)))

    def update_counter2(self, frame2):
        """Helper function for updating slider and frame counter elements"""
        self.window.Element("2 slider").Update(value=frame2)
        time_iteration = self.otschet_v_time(frame2, self.fps2)
        self.window.Element("2 counter").Update("{}/{}".format(time_iteration, self.otschet_v_time(self.frames2, self.fps2)))
    
    def draw_figure(self, canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=True)
        return figure_canvas_agg
    
                
class MyVideoCapture1:
    def __init__(self, video_source1):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source1)
        if not self.vid1.isOpened():
            raise ValueError("Unable to open video source", video_source1)
        # Get video source width and height
        self.width1 = self.vid1.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height1 = self.vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames1 = self.vid1.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps1 = self.vid1.get(cv2.CAP_PROP_FPS)
        self.duration1 = self.frames1/self.fps1

    def get_frame1(self):
        if self.vid1.isOpened():
            ret1, frame1 = self.vid1.read()
            if ret1:
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None
        else:
            return 0, None

    def goto_frame1(self, frame_no1):
        
        if self.vid1.isOpened():
            self.vid1.set(cv2.CAP_PROP_POS_FRAMES, frame_no1)  # Set current frame
            ret1, frame1 = self.vid1.read()  # Retrieve frame
            if ret1:
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None
        else:
            return 0, None
    
    def __del__(self):
        if self.vid1.isOpened():
            self.vid1.release()


class MyVideoCapture2:
    def __init__(self, video_source2):
        # Open the video source
        self.vid2 = cv2.VideoCapture(video_source2)
        if not self.vid2.isOpened():
            raise ValueError("Unable to open video source", video_source2)
        # Get video source width and height
        self.width2 = self.vid2.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height2 = self.vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames2 = self.vid2.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps2 = self.vid2.get(cv2.CAP_PROP_FPS)
        self.duration2 = self.frames2/self.fps2

    def get_frame2(self):
        if self.vid2.isOpened():
            ret2, frame2 = self.vid2.read()
            if ret2:
                return ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                return ret2, None
        else:
            return 0, None

    def goto_frame2(self, frame_no2):
        if self.vid2.isOpened():
            self.vid2.set(cv2.CAP_PROP_POS_FRAMES, frame_no2)  # Set current frame
            ret2, frame2 = self.vid2.read()  # Retrieve frame
            if ret2:
                return ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                return ret2, None
        else:
            return 0, None

    def __del__(self):
        if self.vid2.isOpened():
            self.vid2.release()


class MyDataEEG:

    def __init__(self, data_path):
        
        if data_path[-3:] =='EDF':
            self.data_EEG = mne.io.read_raw_edf(data_path, encoding='latin1')
            self.all_ch_names = self.data_EEG.ch_names
            self.labels_info, self.sampling_frequency = self.get_labels_and_sampling_freq_EDF_plus(self.data_EEG)
            self.channel_data = self.data_EEG.get_data()
            self.data_shape = self.channel_data.shape[1]
            # self.data_value=61000
            self.disabled=False
            self.disabled_EEG=True

        if data_path[-3:] =='edf':
            self.data_EEG = mne.io.read_raw_edf(data_path, encoding='latin1')
            self.all_ch_names = self.data_EEG.ch_names
            self.labels_info, self.sampling_frequency = self.get_labels_and_sampling_freq_EDF_plus(self.data_EEG)
            self.channel_data = self.data_EEG.get_data()
            self.data_shape = self.channel_data.shape[1]
            # self.data_value=61000
            self.disabled=False
            self.disabled_EEG=True

        if data_path[-3:] == 'fif':
            self.data_EEG = mne.io.read_raw_fif(data_path, verbose = False)
            self.all_ch_names = self.data_EEG.ch_names
            self.sampling_frequency = self.data_EEG.info["sfreq"]
            self.channel_data = self.data_EEG.get_data()
            self.data_shape = self.channel_data.shape[1]
            self.disabled=True
            self.disabled_EEG=True
            # self.data_value=34000
            events, events_id = mne.events_from_annotations(self.data_EEG)
            key_list = list(events_id.keys())
            valye_list = list(events_id.values())
            self.data_metki_otschet = events[:,0]
            data_metki_label_1 = events[:,2]
            self.data_metki_label = []
            for metka in data_metki_label_1:
                for i in range(len(key_list)):
                    if metka == valye_list[i]:
                        self.data_metki_label.append(key_list[i])
    
    def get_labels_and_sampling_freq_EDF_plus(self, edf_plus_data):
        sampling_req = edf_plus_data._raw_extras[0]['max_samp']
        label_names = edf_plus_data._annotations.description
        unique_labels = np.unique(edf_plus_data._annotations.description).astype('str')
        label_times = edf_plus_data._annotations.onset*sampling_req
        label_durations = edf_plus_data._annotations.duration*sampling_req
        all_labels_dict = {}
        for i in range(unique_labels.shape[0]):
            used_label = unique_labels[i]
            label_ids = np.where(label_names==used_label)[0]
            labels_starts = label_times[label_ids]
            label_duration = label_durations[label_ids]
            label_amplitude = np.repeat(i+1,label_ids.shape[0])
            name = str(used_label)
            label_end = label_duration+labels_starts
            label_dict = {'label_start':labels_starts, 'label_duration':label_duration,'label_end':label_end,'label_amplitude':label_amplitude}
            all_labels_dict.update({name:label_dict})
        return all_labels_dict, sampling_req



if __name__ == '__main__':
    App()