#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__='Ji Xuanfan'

import os
import re
import json
import time
import urllib
import requests
from matplotlib import pyplot as plt

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


plt.rcParams['font.family']='SimHei'

class Get_Map():
	def __init__(self):
		self.homepage='http://xzqh.mca.gov.cn/map'  #首页的URL
		self.base_url='http://xzqh.mca.gov.cn/defaultQuery?'
		self.headers={'User-Agent': 'Chrome/87.0.4280.88'}

	def Get_all_codes(self,fileName='all_codes.csv'):
		'''
			@function: 获取当前所有行政区划的代码，并存入Data/
			@param:
				fileName: 输出文件的路径及名称（最好取默认值）
		'''
		get_all_district_text=re.compile(r'hidden" value=\'\[(.+?)\]')
		get_all_districts=re.compile(r'cName":"(.+?)","code":"(.+?)","py":"(.+?)","jp":"(.+?)","qp":"(.+?)"')
		res=requests.get(self.homepage,headers=self.headers)
		if not os.path.exists('Data'):
			os.mkdir('Data/')
		if res.status_code==200:
			html=res.text
			temp=re.search(get_all_district_text,html)[1]
			if temp:
				temp=temp.replace('&quot;','"')
				codes_json=re.findall(get_all_districts,temp)
				_dump_csv(codes_json,'Data/'+fileName)
				self._parse_codes()
			else:
				print('数据解析失败')
		else:
			print('网络连接错误')

	def _parse_codes(self,src=r'../Data/all_codes.csv'):
		'''从all_codes.csv提取出省份和地级市的信息，构建dict，key为编码'''
		province={}
		city={}
		code_map={}
		if os.path.exists(src):
			data=_load_csv(src)
			for x in data:
				code_map[x[1]]=x[0]
				if x[1][2:]=='0000':
					province[x[1]]=x[0]
				elif x[1][4:]=='00':
					city[x[1]]=x[0]
			_dump_json(code_map,'../Data/code_map.txt')
			_dump_json(province,'../Data/province.txt')
			_dump_json(city,'../Data/city.txt')
		else:
			print('未获取行政区划的编码~')

	def Get_country(self):
		'''获取首页全国的地图数据'''
		print('正在获取全国地图数据~')
		chrome_options = webdriver.ChromeOptions() #初始化selenium
		prefs = {
			"profile.managed_default_content_settings.images":2,
			"permissions.default.stylesheet": 2,
		}
		chrome_options.add_experimental_option("prefs",prefs)
		chrome_options.set_headless()
		browser = webdriver.Chrome(chrome_options=chrome_options)
		wait = WebDriverWait(browser, 10)  #显式等待

		try:
			browser.get(self.homepage)
			wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#images')))
			html=browser.page_source

			# 将数据存入本地
			save_path='../Data/Country/'
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			print('正在解析全国数据~')
			self.__parse_html_svgMap(html,save_path)
			self.__parse_html_notes(html,save_path)
			self.__parse_html_city_pos(html,save_path)
			self.__parse_html_others(html,save_path)
		except TimeoutException:
			print('等待超时！无法获取全国地图数据~')
		finally:
			browser.quit()

	def Get_provinces_auto(self):
		'''自动获取尚未获取的省份数据（不包括港澳台）'''
		if not os.path.exists(r'../Data/Province/'):
			os.mkdir(r'../Data/Province/')
		has_got=set([x[:2] for x in os.listdir(r'../Data/Province')]+['71','81','82'])

		if not os.path.exists(r'../Data/province.txt'):
			print('省份编码(province.txt)文件不存在~')
			return
		all_provinces=_load_json(r'../Data/province.txt')

		need_to_get=[x[:2] for x in all_provinces.keys() if x[:2] not in has_got]
		self.Get_provinces(need_to_get)

	def Get_provinces(self,codes):
		'''
			@func:获取多个省份的地图数据（主要是省内各个地级行政单位）
			@para:
				codes:list,省份的二位编码，不含港澳台地区
		'''
		codes=list(map(str,codes))

		if not os.path.exists(r'C:\Users\MSZ\Desktop\GCRNN_PyTorch\Map_of_China\Data\province_code.csv'):
			print('省份编码(province_code.csv)文件不存在~')
			return
		if not os.path.exists(r'C:\Users\MSZ\Desktop\GCRNN_PyTorch\Map_of_China\Data\Province'):
			os.mkdir(r'C:\Users\MSZ\Desktop\GCRNN_PyTorch\Map_of_China\Data\Province/')

		province_code=_load_csv(r'C:\Users\MSZ\Desktop\GCRNN_PyTorch\Map_of_China\Data\province_code.csv')
		province={x[0]:[x[2],x[3]] for x in province_code}
		del province['71'],province['81'],province['82']  #删除港澳台地区
		names=[]
		for x in codes:
			if x not in province:
				print('无法获取编码为%s省份的数据'%x)
			else:
				names.append([province[x][0],x+'-'+province[x][1]])  #用于查询的省份名，输出文件的名称

		chrome_options = webdriver.ChromeOptions() #初始化selenium
		prefs = {
			"profile.managed_default_content_settings.images":2,
			"permissions.default.stylesheet": 2,
		}
		chrome_options.add_experimental_option("prefs",prefs)
		chrome_options.set_headless()
		browser = webdriver.Chrome(chrome_options=chrome_options)
		wait = WebDriverWait(browser, 10)  #显式等待

		for name in names:
			self.__get_province(name[0],name[1],browser,wait)
			time.sleep(2)
		browser.quit()

	def __get_province(self,name,dump_file,browser,wait):
		'''获取某个省内各地级市的数据'''
		name_encoded = urllib.parse.quote(name,encoding='gbk')
		url=self.base_url+'shengji=%s&diji=-1&xianji=-1'%name_encoded
		print('正在获取%s的数据'%(name))
		try:
			browser.get(url)
			wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#others')))
			html=browser.page_source
			self.__parse_province_html(html,dump_file)
		except TimeoutException:
			print('等待超时，无法获取%s的数据'%name)

	def __parse_province_html(self,html,dump_file):
		'''解析省份的HTML数据'''
		save_path='../Data/Province/%s/'%(dump_file)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		self.__parse_html_svgMap(html,save_path)
		self.__parse_html_notes(html,save_path)
		self.__parse_html_city_pos(html,save_path)
		self.__parse_html_others(html,save_path)

	def Get_cities_auto(self,code=None):
		'''
			@func:自动获取尚未获取的地级行政单位的地图数据
			@code:
				None, 获取全部未获取的地级行政单位
				int/str,2位省份编码，获取某个省内未获取的地级行政单位数据
		'''
		if not os.path.exists('../Data/city.txt'):
			print('地级行政单位编码文件(city.txt)不存在')
			return
		if not os.path.exists('../Data/City/'):
			os.mkdir('../Data/City/')
		has_got=[x.split('-')[0]+'00' for x in os.listdir('Data/City/')]
		need_to_get=[]
		if code:
			code=str(code)
			if len(code)!=2:
				print('省份编码不是两位数~')
				return
		all_codes=_load_json('../Data/city.txt')
		for k,v in all_codes.items():
			if k not in has_got:
				if code and k[:2]!=code:
					continue
				need_to_get.append(k[:4])

		self.Get_cities(need_to_get)

	def Get_cities(self,codes):
		'''
			@func:获取多个地级市的数据
			@para:
				codes:list,地级市的四位编码，不含直辖市、港澳台地区和三沙市（460300）
		'''
		codes=list(map(str,codes))
		codes=[x.ljust(6,'0') for x in codes]
		# 检验省份和地级市的编码文件是否存在
		if not os.path.exists('../Data/province_code.csv'):
			print('省份编码(province_code.csv)文件不存在~')
			return
		if not os.path.exists('Data/city.txt'):
			print('地级市编码(city.csv)文件不存在~')
			return
		if not os.path.exists('../Data/City/'):
			os.mkdir('../Data/City/')

		province_code=_load_csv('../Data/province_code.csv')
		province={x[0]:x[2] for x in province_code}  #省份2位编码:省份全称(简称)
		city_code=_load_json('../Data/city.txt')
		del city_code['460300'] #删除三沙市
		names=[]
		for x in codes:
			if x not in city_code:
				print('无法获取编码为%s地级市的数据'%x)
			else:
				names.append([province[x[:2]],city_code[x],x[:4]])

		chrome_options = webdriver.ChromeOptions() #初始化selenium
		prefs = {
			"profile.managed_default_content_settings.images":2,
			"permissions.default.stylesheet": 2,
		}
		chrome_options.add_experimental_option("prefs",prefs)
		chrome_options.set_headless()
		browser = webdriver.Chrome(chrome_options=chrome_options)
		wait = WebDriverWait(browser, 10)  #显式等待

		for name in names:
			print('正在获取%s-%s的数据'%(name[0],name[1]))
			self.__get_city(name[0],name[1],name[2],browser,wait)
			time.sleep(2)
		browser.quit()

	def __get_city(self,shengji,diji,code,browser,wait):
		'''获取某地级市内各县级市的数据'''
		shengji_encoded = urllib.parse.quote(shengji,encoding='gbk')
		diji_encoded=urllib.parse.quote(diji,encoding='gbk')
		url=self.base_url+'shengji=%s&diji=%s&xianji=-1'%(shengji_encoded,diji_encoded)
		try:
			browser.get(url)
			wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#others')))
			html=browser.page_source
			self.__parse_city_html(html,code+'-'+diji)
		except TimeoutException:
			print('等待超时，无法获取%s-%s的数据'%(shengji,diji))

	def __parse_city_html(self,html,dump_file):
		'''解析地级市的HTML数据'''
		save_path='Data/City/%s/'%(dump_file)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		self.__parse_html_svgMap(html,save_path)
		self.__parse_html_notes(html,save_path)
		self.__parse_html_city_pos(html,save_path)
		self.__parse_html_others(html,save_path)

	def Get_country_county(self):
		'''
			@func: 在全国层面获取县级数据
		'''
		print('正在获取全国的县级数据')
		chrome_options = webdriver.ChromeOptions() #初始化selenium
		prefs = {
			"profile.managed_default_content_settings.images":2,
			"permissions.default.stylesheet": 2,
		}
		chrome_options.add_experimental_option("prefs",prefs)
		chrome_options.set_headless()
		browser = webdriver.Chrome(chrome_options=chrome_options)
		wait = WebDriverWait(browser, 20)  #显式等待

		try:
			browser.get(self.homepage)
			browser.find_element_by_id("xianshixianji").click()
			time.sleep(8)  #网页动态生成地图，需要几秒时间
			wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#svgMap')))
			html=browser.page_source

			# 将数据存入本地
			print('正在解析全国数据~')
			self.__parse_country_county(html)
		except TimeoutException:
			print('等待超时！无法获取全国地图数据~')
		finally:
			browser.quit()

	def Get_South_China_Sea(self):
		'''
			@func: 获取中国南海的缩略图数据，自动提取出九段线数据，存入sea_path.txt
		'''
		if not os.path.exists('Data/Country'):
			os.mkdir('Data/Country/')
		chrome_options = webdriver.ChromeOptions() #初始化selenium
		prefs = {
			"profile.managed_default_content_settings.images":2,
			"permissions.default.stylesheet": 2,
		}
		chrome_options.add_experimental_option("prefs",prefs)
		chrome_options.set_headless()
		browser = webdriver.Chrome(chrome_options=chrome_options)
		wait = WebDriverWait(browser, 10)  #显式等待

		name_encoded = urllib.parse.quote('海南省(琼)',encoding='gbk')
		url=self.base_url+'shengji=%s&diji=-1&xianji=-1'%name_encoded
		try:
			browser.get(url)
			wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#others')))
			html=browser.page_source
			self.__parse_nanhai_html(html)
		except TimeoutException:
			print('等待超时，无法获取南海地图的数据')
		finally:
			browser.quit()

	def __parse_nanhai_html(self,html):
		'''解析南海地图数据'''
		if not os.path.exists('Data/Country/Sea'):
			os.mkdir('Data/Country/Sea/')
		save_path='Data/Country/Sea/'

		print('正在解析南海地图数据~')
		re_nanhai_txt=re.compile(r'id="nanhai".+?>\n(.+?)</svg></div>')
		html=re_nanhai_txt.search(html)[1]

		self.__parse_html_others(html,save_path)

		data=_load_json('Data/Country/Sea/others.txt')
		out_data=[]
		for x in data:
			val=[[float(y)/2 for y in t.split(',')] for t in x[1:].split('L')]
			X_pos=[t[0]+720 for t in val]
			Y_pos=[t[1]+420 for t in val]
			out_data.append([X_pos,Y_pos])
		out_data=[out_data[0]]+out_data[14:17]+out_data[73:82]+out_data[97:101]
		_dump_json(out_data,'Data/Country/Sea/sea_path.txt')

	def __parse_html_svgMap(self,html,save_path):
		'''解析HTML中的形状数据（包括形状的颜色）'''
		get_svgMap_text=re.compile(r'svgMap">(.+?)</g>')
		get_path_text=re.compile(r'<path(.+?)>')
		get_svgMap=re.compile(r'd="(.+?)" id="(.+?)"(.+?)style="(.+?)"')
		get_svgMap_bg=re.compile(r'rgb\((\d+?), (\d+?), (\d+?)\)')

		temp=re.search(get_svgMap_text,html)[1]
		path=[x for x in re.findall(get_path_text,temp) if ' d=' in x]
		shapes=[]
		for x in path:
			temp=re.search(get_svgMap,x)
			if temp:
				shapes.append([temp[1],temp[2],temp[4]])
		for i in range(len(shapes)):
			style=shapes[i][2]
			temp=re.search(get_svgMap_bg,style)
			if temp:
				shapes[i][2]=(int(temp[1]),int(temp[2]),int(temp[3]))
			else:
				shapes[i][2]=None
		output={}
		for x in shapes:
			if x[1] in output:
				output[x[1]].append([x[0],x[2]])
			else:
				output[x[1]]=[[x[0],x[2]]]
		for k,v in output.items():
			_dump_json(v,save_path+k+'.txt')

	def __parse_html_notes(self,html,save_path):
		'''解析注释的文字'''
		get_notes_text=re.compile(r'notes.+?>(.+?)</g></g')
		get_notes=re.compile(r'<g id="(.+?)" font-size="(\d+?)px.+?dy="(.+?)".+?fill: (.+?);.+?>(.+?)</g>')
		get_notes_pos=re.compile(r'<text x="(.+?)" y="(.+?)".*?>(.*?)</text>')

		temp=re.search(get_notes_text,html)
		if temp:
			temp=temp.group(1)+'</g>'
			notes=[list(x) for x in re.findall(get_notes,temp)]
			for i in range(len(notes)):
				text=notes[i][4]
				notes[i][4]=re.findall(get_notes_pos,text)
			_dump_json(notes,save_path+'notes.txt')
		else:
			print('无法解析注释中的文字')

	def __parse_html_city_pos(self,html,save_path):
		'''解析主要城市的位置和名称'''
		get_city_pos_text=re.compile(r'images">(.+?)</g>')
		get_city_pos=re.compile(r'x="(.+?)" y="(.+?)"')
		get_city_text=re.compile(r'texts">(.+?)</g>')
		get_city=re.compile(r'x="(.+?)" dx="(.+?)" y="(.+?)" dy="(.+?)".+?font-size: (\d+?)px; text-anchor: (.+?);.+?>(.+?)</')

		temp=re.search(get_city_pos_text,html)
		if temp:
			city_pos=re.findall(get_city_pos,temp[1]) #城市的位置坐标

			temp2=re.search(get_city_text,html)[1]
			cities=re.findall(get_city,temp2)

			out=list(zip(city_pos,cities))
			_dump_json(out,save_path+'city_pos.txt')
		else:
			print('无法解析主要城市的位置和名称')

	def __parse_html_others(self,html,save_path):
		'''解析其他图形数据'''
		get_others_text=re.compile(r'others">(.*?)</g>')
		get_others=re.compile(r'd="(.+?)"')

		temp=re.search(get_others_text,html)
		if temp:
			temp=temp.group(1)
			others=re.findall(get_others,temp)
			_dump_json(others,save_path+'others.txt')
		else:
			print('无法解析Others字段数据')

	def __parse_country_county(self,html):
		'''解析在全国层面获取的县级数据'''
		re_path_text=re.compile(r'<path stroke.+?d="(.+?)" id="(.+?)"')  #路径和相应的区域编码
		data=re_path_text.findall(html)
		outVal={}  #key为区域编码，val为路径坐标：list,[[[x1,x2,...],[y1,y2,...]],[...]]
		for x in data:
			if x[1] in outVal.keys():
				outVal[x[1]]+=self.parse_path(x[0])
			else:
				outVal[x[1]]=self.parse_path(x[0])
		_dump_json(outVal,path='Data/country_county.txt')

	def parse_path(self,path):
		'''
			@func: 将单个路径的SVG指令字符串转换成坐标
			@path:
				str，例如"M..L..L..ZM..L..L..Z"这样的SVG指令
			@output:
				list, 每个元素为list，该list有两个元素，第一个为X轴坐标，第二个为Y轴坐标
		'''
		outVal=[]
		parts=path.split('M')[1:]
		for part in parts:
			pos=[[float(y) for y in t.split(',')] for t in part[:-1].split('L')]
			X_pos=[t[0] for t in pos]
			Y_pos=[t[1] for t in pos]
			X_pos.append(X_pos[0])
			Y_pos.append(Y_pos[0])
			outVal.append([X_pos,Y_pos])
		return outVal

class Draw_Map():
	def __init__(self):
		self.draw_country_arg={'sea':False,'provName':False,'textColor':'r','default_color':[0.9,0.9,0.9],\
							'city_pos':False,'point_color':'r','point_size':3,'city_name':False,'cityTextColor':'k'}
		self.draw_region_arg={'textColor':'k','default_color':[0.9,0.9,0.9],'thre':None,
							'other':False,'other_lw':0.5,'city_pos':False,'point_color':'r','point_size':3,
							'city_name':False}
		self.pos_map={'start':'left','middle':'center','end':'right'}  #城市名称注释的位置与matplotlib.text 中ha的对应关系
		self.country_county_arg={'island':True,'island_color':[0,0.43,0.74]}


	def Draw_single_shape(self,path,LC='w',FC=[30,120,180],BC='w',lw=1,title=None,img_path=None):
		'''
			@function
				绘制单一路径的形状或轮廓，如一个省、市的轮廓线
			@param
				path: str,路径文件位置
				LC: 线条颜色
					str,颜色简称，如r/w/k/g/b
					str,十六进制颜色
					list,RGB格式的颜色，0-255
					None,不添加线条
				FC: 填充颜色
					None，不填充颜色；其他参数格式同LC
				BC: 背景色
				lw: 线宽
				title: 图片标题
				img_path: 图片保存路径
						str, 相对路径
						None, 直接显示图片，不保存
			@example
				D=Draw_Map()
				D.Draw_single_shape('Data/Country/320000.txt',LC=None)  #江苏地图
				D.Draw_single_shape('Data/Province/32-江苏/321100.txt',LC=None,title='江苏镇江')  #江苏镇江地图
		'''
		if not os.path.exists(path):
			print('图形文件不存在！')
			return
		# 设置颜色
		if LC:
			if isinstance(LC,list) or isinstance(LC,tuple):
				LC=[x/255 for x in LC]  #线条颜色
		if FC:
			if isinstance(FC,list) or isinstance(FC,tuple):
				FC=[x/255 for x in FC]  #填充颜色
		if BC:
			if isinstance(BC,list) or isinstance(BC,tuple):
				BC=[x/255 for x in BC] #背景颜色

		raw_data=_load_json(path)  #原始的路径数据，list
		add_path=[]  #需要绘制的路径
		del_path=[]  #需要删除的路径，主要是飞地
		for x in raw_data:
			data=x[0].split('M')[1:]
			if len(data)==1:
				temp=[[float(y) for y in t.split(',')] for t in data[0][:-1].split('L')]  #每一个点的坐标
				add_path.append(temp)
			elif len(data)>1:
				sorted_data=sorted(data,key=len,reverse=True)
				if len(sorted_data[1])/len(sorted_data[0])<0.5:
					#如何最长的路径和次长的路径长度差异很大，则认为较短的路径为其他区域的飞地
					temp=[[float(y) for y in t.split(',')] for t in sorted_data[0][:-1].split('L')]
					add_path.append(temp)
					for d_data in sorted_data[1:]:
						d=[[float(y) for y in t.split(',')] for t in d_data[:-1].split('L')]
						del_path.append(d)
				else:
					for add_data in sorted_data:
						a=[[float(y) for y in t.split(',')] for t in add_data[:-1].split('L')]
						add_path.append(a)
		self._draw_outline(add_path,del_path,LC,FC,BC,lw,title,img_path)

	def _draw_outline(self,add_path,del_path=[],LC=[1,1,1],FC=[0.12,0.47,0.71],BC=[1,1,1],lw=1,title=None,img_path=None):
		plt.figure(facecolor=BC)
		for d in add_path:
			X=[t[0] for t in d]
			Y=[t[1] for t in d]
			X.append(X[0])
			Y.append(Y[0])
			if LC:
				plt.plot(X,Y,color=LC,linewidth=lw)
			if FC:
				plt.fill(X,Y,color=FC)
		for d in del_path:
			X=[t[0] for t in d]
			Y=[t[1] for t in d]
			X.append(X[0])
			Y.append(Y[0])
			if LC:
				plt.plot(X,Y,color=LC,linewidth=lw)
			if FC:
				plt.fill(X,Y,color=BC)

		if title:
			plt.title(title)
		plt.gca().set_aspect("equal")
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.gcf().tight_layout()
		if img_path:
			plt.savefig(img_path,facecolor=BC)
			plt.cla()
		else:
			plt.show()

	def Show_data(self,src,code=None,by_val=True,LC='k',FC='Blues',BC='w',title=None,img_path=None,arg={}):
		'''
		@function：绘制热力图（全国、省级、地级），即将值的大小与区域颜色深浅对应
		@param:
			src: 数据文件的路径，第一列为区域名称或者编码（要么全部是名称，要么全部是编码），第二列为数值
			code：
				None： 绘制全国地图
				str: 2位编码，绘制相应的省份地图；4位编码，绘制相应的地级行政单位地图
			by_val：
				True：按照值的绝对大小显示
				False：按照值的相对大小显示（排名）
			LC：区域边框的颜色
				str, 如'r','#123456'
				list, RGB格式，0-255范围
			FC：区域的填充颜色
				str：matplotlib支持的cmap颜色字符串，默认Blues
				str：如'r','#ff0000'（十六进制）
				list, RGB格式，0-255范围
			BC：背景色，参数含义同LC
			title：图片标题（默认None，不显示标题）
			img_path：图片的输出存储路径（包括名称）
				None：直接在屏幕中显示图片
			arg: dict,图片参数的配置，具体与Draw_country、Draw_province、Draw_city保持一致
		@ Example:
			D=Draw_Map()
			src='Example/GDP-2018.csv'
			D.Show_data(src,FC='PuRd',title='2018年各省GDP',arg={'sea':True})  #按数据的绝对大小显示
			D.Show_data(src,by_val=False,arg={'default_color':[0.6,0.6,0.6]})  #按数据的相对大小(排名)显示

			D.Show_data('Example/江苏人口2018.txt','32',by_val=True,title='江苏人口2018') #省级数据
			D.Show_data('Example/江苏人口2018.txt','32',by_val=True,LC='w',\
				title='江苏人口2018',arg={'city_name':True,'city_pos':True,'other':True}) #可配置地图中的样式
			
			D.Show_data('Example/Suzhou_population.txt','3205',by_val=False,\
				title='苏州人口2018',arg={'city_name':True,'city_pos':True,'other':True}) #地级数据
		'''
		if not os.path.exists(src):
			print('数据文件不存在！')
			return
		else:
			raw_data=_load_csv(src)
		if not isinstance(FC,str):
			print('FC不是字符串类型，不是合格的cmap参数')
			return
		if not code:
			level=1
		elif len(code)==2 or len(code)==4:
			level=len(code)//2+1
		else:
			print('code编码格式不正确：非2位省份编码或4位地级市编码')
			return
		if code: code=str(code)
		data={}
		if level==1:  #在国家层面比较各个省的情况
			if not os.path.exists('Data/province.txt'): #导入省份的编码文件
				print("省份编码文件不存在")
				return
			code_map=_load_json('Data/province.txt')
		elif level==2 or level==3:  #省份2位编码或地级市4位编码
			all_codes=_load_json('Data/code_map.txt')
			code_map={}
			L=len(code)
			for k,v in all_codes.items():
				if k[:L]==code and k[L:L+2]!='00':
					code_map[k]=v

		if raw_data[0][0].isdigit():
			for x in raw_data:
				if len(x[0])%2==0 and len(x[0])<=6:
					key=x[0].ljust(6,'0')
				else:
					print(x[0]+'不是正确的区域编码')
					return
				if key in code_map.keys():
					data[key]=float(x[1])
				else:
					print(x[0]+'不是正确的区域名称')
					return
		else:
			for x in raw_data:
				has_got=False  #标记是否已查询到区域编码
				for k,v in code_map.items():
					if x[0] in v:
						data[k]=float(x[1])
						has_got=True
						break
				if not has_got:
					print(x[0]+'不是正确的区域名称')
					return

		if level==1:
			self.Draw_country(data,by_val=by_val,LC=LC,FC=FC,BC=BC,title=title,img_path=img_path,arg=arg)
		elif level==2:
			self.Draw_province(code,data,by_val=by_val,LC=LC,FC=FC,BC=BC,title=title,img_path=img_path,arg=arg)
		elif level==3:
			self.Draw_city(code,data,by_val=by_val,LC=LC,FC=FC,BC=BC,title=title,img_path=img_path,arg=arg)

	def Draw_country(self,data=None,by_val=True,LC='k',FC=[30,120,180],BC='w',lw=1,title=None,img_path=None,arg={}):
		'''
			@function：绘制全国地图（给不同省份标上不同颜色，可配置省份名称、城市名称、南海区域样式等）
			@param:
				data: 直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				by_val：直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				LC：区域边框的颜色
					str, 如'r','#123456'
					list, RGB格式，0-255范围
				FC：区域的填充颜色
					True: 默认采用民政部官网中的颜色
					str：如'r','#ff0000'（十六进制）
					list, RGB格式，0-255范围
				BC：背景色，参数含义同LC
				lw: 区域边框的线宽
				title：图片标题（默认None，不显示标题）
				img_path：图片的输出存储路径（包括名称）
					None：直接在屏幕中显示图片
				arg: dict,图片参数的配置
					sea: 
						False: 绘制缩略的南海地图（默认）
						True： 绘制展开的南海地图
					provName：是否显示省份的名称
					textColor：省份名称的颜色，默认红色
					default_color：当数据缺失时区域的颜色
					city_pos：是否绘制省会城市所在位置的点（当该参数为False时，city_name无效）
					point_color：绘制省会城市所在位置的点的半径
					city_name：是否显示省会城市的名称（默认否）
					cityTextColor：省会城市名称文字的颜色（默认黑色）						
		'''
		# 设置绘图选项
		for k,v in arg.items():
			if k in self.draw_country_arg:
				self.draw_country_arg[k]=v

		if not os.path.exists('Data/Country'):
			print('省份图形文件不存在~')
			return
		if not os.path.exists('Data/province.txt'):
			print('省份编码文件不存在')
			return
		else:
			province_code=_load_json('Data/province.txt')
		province_shape_file=[x for x in os.listdir('Data/Country/') if len(x)==10 and x[-5]=='0'] #各省图形文件

		sea_path=[]  #完整的南海诸岛地图数据
		if self.draw_country_arg['sea']:  #是否绘制南海诸岛
			if not os.path.exists('Data/Country/daodian.txt'):
				print('daodian.txt文件缺失，无法绘制岛屿~')
				return  #地图中必须包含南海诸岛
			else:
				province_shape_file.append('daodian.txt')
				if not os.path.exists('Data/Country/others.txt'):
					print('others.txt文件缺失，无法绘制九段线')
				else:
					temp_nine_dotted=sorted(_load_json('Data/Country/others.txt'),key=len)[:14]
					for x in temp_nine_dotted:
						val=[[float(y) for y in t.split(',')] for t in x[1:].split('L')]
						sea_path.append(val)
		else:
			if not os.path.exists('Data/Country/Sea/sea_path.txt'):
				print('南海地图数据缺失~')
				return
			else:
				sea_path=_load_json('Data/Country/Sea/sea_path.txt')

		provName=None
		if self.draw_country_arg['provName']:  #是否标注省份名称
			if not os.path.exists('Data/Country/notes.txt'):
				print('省份名称文件不存在!')
			else:
				temp_name_data=_load_json('Data/Country/notes.txt')
				provName=[]
				for x in temp_name_data:
					if x[0].isdigit():
						provName.append([x[1],x[4]])

		# 根据data的大小或排名计算颜色值
		color_map={}  #省份对应的颜色
		if data and isinstance(data,dict):
			if isinstance(FC,str):
				color_map=self.__get_cmap(data,FC,by_val)
			else:
				print('参数FC不是cmap值！')
				return

		# 设置颜色
		LC,FC,BC=self.__set_color([LC,FC,BC])
		plt.figure(facecolor=BC) #背景色

		for x in province_shape_file:
			parts=_load_json('Data/Country/'+x)
			for part in parts:
				shape=part[0][1:-1]
				pos=[[float(y) for y in t.split(',')] for t in shape.split('L')]
				X_pos=[t[0] for t in pos]
				Y_pos=[t[1] for t in pos]
				X_pos.append(X_pos[0])
				Y_pos.append(Y_pos[0])
				if LC:
					plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
				if FC:
					if FC is True:
						color=part[1]
						color=[color[0]/255,abs(color[1]-10)/255,abs(color[2]-30)/255]
					elif color_map:
						if x[:6] in color_map:
							color=color_map[x[:6]]
						else:
							color=self.draw_country_arg['default_color']  #数据缺失时默认颜色
					else:
						color=FC
					plt.fill(X_pos,Y_pos,color=color)

		# 绘制南海诸岛
		if self.draw_country_arg['sea']:
			for pos in sea_path:
				X_pos=[t[0] for t in pos]
				Y_pos=[t[1] for t in pos]
				plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
		else:
			for pos in sea_path:
				X_pos=pos[0]
				Y_pos=pos[1]
				plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
			plt.plot([740,855,855,740,740],[415,415,570,570,415],color=LC,linewidth=lw*1.2)

		# # 添加省份名称
		fontColor=self.draw_country_arg['textColor']
		if provName:
			for x in provName:
				fz=int(x[0])-1   #字号
				pos=x[1]  #文字坐标
				for y in pos:
					plt.text(float(y[0]),float(y[1]),y[2],ha='center',va='center',fontsize=fz,color=fontColor)

		# 添加主要城市的名称
		if self.draw_country_arg['city_pos']:
			if os.path.exists('Data/Country/city_pos.txt'):
				city_pos=_load_json('Data/Country/city_pos.txt')
				X_pos,Y_pos=[],[]
				for x in city_pos:
					X_pos.append(float(x[0][0])+5) #点的坐标
					Y_pos.append(float(x[0][1])+5)
					if self.draw_country_arg['city_name']:
						plt.text(float(x[1][0])+float(x[1][1])+5,float(x[1][2])+float(x[1][3])+5,x[1][6],fontsize=int(x[1][4]),\
							color=self.draw_country_arg['cityTextColor'],ha=self.pos_map[x[1][5]])
				plt.plot(X_pos,Y_pos,'o',color=self.draw_country_arg['point_color'],\
					markersize=self.draw_country_arg['point_size'])

		if title:
			plt.title(title)
		plt.gca().set_aspect("equal")
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.gcf().tight_layout()
		if img_path:
			plt.savefig(img_path,facecolor=BC)
			plt.cla()
		else:
			plt.show()

	def Draw_province(self,code,data=None,by_val=True,LC='k',FC=[30,120,180],BC='w',lw=1,title=None,img_path=None,arg={}):
		'''
			@func: 绘制某个省内各地级市
			@param:
				code: 省份的2位编码或者名称(简称、全称)
				data: 直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				by_val：直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				LC：区域边框的颜色
					str, 如'r','#123456'
					list, RGB格式，0-255范围
				FC：区域的填充颜色
					True: 默认采用民政部官网中的颜色
					str：如'r','#ff0000'（十六进制）
					list, RGB格式，0-255范围
				BC：背景色，参数含义同LC
				lw: 区域边框的线宽
				title：图片标题（默认None，不显示标题）
				img_path：图片的输出存储路径（包括名称）
					None：直接在屏幕中显示图片
				arg: dict,图片参数的配置
					textColor：地级市名称的颜色，默认黑色
					default_color：当数据缺失时区域的颜色
					thre：
						 None：默认绘制全部地图数据
						 int：区域路径的最短长度（一些省份有很多岛屿，导致绘图较慢；
						 	当不想显示太小的岛屿，可以将该参数调大）
					other：是否显示周围省份或城市的地图
					other_lw：当other为True时，设定区域边框的线宽
					city_pos：是否绘制地级行政单位城市所在位置的点（当该参数为False时，city_name无效）
					point_color：地级行政单位城市所在位置的点的颜色
					point_size：地级行政单位城市所在位置的点的半径
					city_name：是否显示地级行政单位城市的名称（默认否）
		'''
		#导入省份编码文件
		province_code=_load_csv('Data/province_code.csv')
		prov_map={x[0]:x[3]  for x in province_code}
		reverse_map={x[2]:x[0]  for x in province_code}
		code=str(code)
		if code.isdigit():
			if code not in prov_map.keys():  #验证编码是否有误
				print('省份编码%s有误'%code)
				return
		else:
			has_got=False
			for k in reverse_map.keys():
				if code in k:
					code=reverse_map[k]
					has_got=True
					break
			if not has_got:
				print('省份名称%s有误'%code)
				return

		path='Data/Province/%s-%s/'%(code,prov_map[code])
		if not os.path.exists(path):
			print('%s-%s的图形文件不存在!'%(code,prov_map[code]))
			return
		self.draw_region(path,data=data,by_val=by_val,LC=LC,FC=FC,BC=BC,lw=lw,title=title,img_path=img_path,arg=arg)

	def Draw_city(self,code,data=None,by_val=True,LC='k',FC=[30,120,180],BC='w',lw=1,title=None,img_path=None,arg={}):
		'''
			@func: 绘制某个省内各地级市
			@param:
				code: 地级市的4位编码
				data: 直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				by_val：直接调用该函数时，忽略该参数。该参数是给Show_data函数调用的，用于绘制热力图
				LC：区域边框的颜色
					str, 如'r','#123456'
					list, RGB格式，0-255范围
				FC：区域的填充颜色
					True: 默认采用民政部官网中的颜色
					str：如'r','#ff0000'（十六进制）
					list, RGB格式，0-255范围
				BC：背景色，参数含义同LC
				lw: 区域边框的线宽
				title：图片标题（默认None，不显示标题）
				img_path：图片的输出存储路径（包括名称）
					None：直接在屏幕中显示图片
				arg: dict,图片参数的配置
					textColor：县级行政区划名称的颜色，默认黑色
					default_color：当数据缺失时区域的颜色
					thre：
						 None：默认绘制全部地图数据
						 int：区域路径的最短长度（一些省份有很多岛屿，导致绘图较慢；
						 	当不想显示太小的岛屿，可以将该参数调大）
					other：是否显示周围省份或城市的地图
					other_lw：当other为True时，设定区域边框的线宽
					city_pos：是否绘制县级行政区划城市所在位置的点（当该参数为False时，city_name无效）
					point_color：县级行政单位城市所在位置的点的颜色
					point_size：县级行政单位城市所在位置的点的半径
					city_name：是否显示县级行政单位城市的名称（默认否）
			@example:
				D.Draw_Map()
				D.Draw_city('3211')
				D.Draw_city('镇江',FC=True,LC='w',arg={'other':True,'city_pos':True,'city_name':True})
		'''
		#导入地级市编码文件
		city_code=_load_json('Data/city.txt')
		code=str(code)
		if code.isdigit():
			code=code.ljust(6,'0')
			if code not in city_code.keys():
				print('地级行政单位编码%s有误'%code)
				return
		else:
			has_got=False
			for k,v in city_code.items():
				if code in v:
					code=k
					has_got=True
					break
			if not has_got:
				print('地级行政单位名称%s有误'%code)
				return

		path='Data/City/%s-%s/'%(code[:4],city_code[code])
		if not os.path.exists(path):
			print('%s-%s的图形文件不存在!'%(code[:4],city_code[code]))
			return
		self.draw_region(path,data=data,by_val=by_val,LC=LC,FC=FC,BC=BC,lw=lw,title=title,img_path=img_path,arg=arg)

	def draw_region(self,path,data=None,by_val=True,LC='k',FC=[30,120,180],BC='w',lw=1,title=None,img_path=None,arg={}):
		'''
			@func: 绘制某个区域内各部分的地图（如省内各地级市，或地级市内各县级市）
					一般不直接调用该函数，而是供Draw_city和Draw_province调用
		'''
		# 设置绘图选项
		for k,v in arg.items():
			if k in self.draw_region_arg:
				self.draw_region_arg[k]=v

		src_data=[x for x in os.listdir(path) if x[:6].isdigit()] #名称为6位区域编码的文件
		regions,other_part=[],[]
		code,region_name=path.split('/')[-2].split('-')
		level=len(code)
		for x in src_data:
			if x[:level]==code and x[level:level+2]!='00':
				regions.append(x)
			elif x[:level]!=code:
				other_part.append(x)
		if level==4 and not regions:
			regions.append(code+'00.txt') #直筒子市

		# 根据data的大小或排名计算颜色值
		color_map={}
		if data and isinstance(data,dict):
			if isinstance(FC,str):
				color_map=self.__get_cmap(data,FC,by_val) #省份对应的颜色
			else:
				print('参数FC不是cmap值！')
				return

		# 设置颜色
		LC,FC,BC=self.__set_color([LC,FC,BC])
		plt.figure(facecolor=BC) #背景色

		draw_data=[]  #保存绘图的数据：FC+多段路径坐标
		for x in regions:
			parts=_load_json(path+x)  # 可能有多个部分
			for part in parts:  #每一部分可能有多段，多余的很可能是飞地
				path_data=part[0].split('M')[1:]  #划分为多段
				if FC is True:
					if part[1] is None:
						continue  #直筒子市可能有重复的数据，第一个颜色为None，会出错
					color=[abs((c-5)/255) for c in part[1]]
				elif color_map:
					if x[:6] in color_map:
						color=color_map[x[:6]]
					else:
						color=self.draw_region_arg['default_color']
				else:
					color=FC
				if len(path_data)==1:  #只有一段路径
					temp=[[float(y) for y in t.split(',')] for t in path_data[0][:-1].split('L')]  #每一个点的坐标
					draw_data.append([color,temp])
				elif len(path_data)>1:
					sorted_data=sorted(path_data,key=len,reverse=True)
					if len(sorted_data[1])/len(sorted_data[0])<0.5:
						temp=[[float(y) for y in t.split(',')] for t in sorted_data[0][:-1].split('L')]
						draw_data.append([color,temp])
					else:
						for add_data in sorted_data:
							a=[[float(y) for y in t.split(',')] for t in add_data[:-1].split('L')]
							draw_data.append([color,a])
		sorted_draw_data=sorted(draw_data,key=lambda x:len(x[1]),reverse=True)
		for x in sorted_draw_data:
			thre=self.draw_region_arg['thre']
			if not thre or (thre and len(x[1])>thre):
				color,pos=x[0],x[1]
				X_pos=[t[0] for t in pos]
				Y_pos=[t[1] for t in pos]
				X_pos.append(X_pos[0])
				Y_pos.append(Y_pos[0])
				if LC:
					plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
				if color:
					plt.fill(X_pos,Y_pos,color=color)

		# 绘制其他部分
		if self.draw_region_arg['other']:
			for x in other_part:
				parts=_load_json(path+x)  # 可能有多个部分
				for part in parts:
					path_data=part[0].split('M')[1:]  #划分为多段
					pos=[[float(y) for y in t.split(',')] for t in path_data[0][:-1].split('L')]  #默认每一部分只有一段
					X_pos=[t[0] for t in pos]
					Y_pos=[t[1] for t in pos]
					X_pos.append(X_pos[0])
					Y_pos.append(Y_pos[0])
					plt.plot(X_pos,Y_pos,color='k',linewidth=self.draw_region_arg['other_lw'])
			if os.path.exists(path+'fanwei.txt'):  #区域的范围
				fanwei=_load_json(path+'fanwei.txt')
				fanwei_path=fanwei[0][0][1:-1].split('L')
				fanwei_pos=[[float(y) for y in x.split(',')] for x in fanwei_path]
				X_pos=[t[0] for t in fanwei_pos]
				Y_pos=[t[1] for t in fanwei_pos]
				X_pos.append(X_pos[0])
				Y_pos.append(Y_pos[0])
				plt.plot(X_pos,Y_pos,color='k',linewidth=self.draw_region_arg['other_lw'])
			if os.path.exists(path+'notes.txt'):
				notes=_load_json(path+'notes.txt')
				for x in notes:
					fontsize=int(x[1])-2
					for y in x[4]:
						plt.text(float(y[0])-10,float(y[1]),y[2],fontsize=fontsize,color='k')

		# 标注城市的坐标
		if self.draw_region_arg['city_pos']:
			if os.path.exists(path+'city_pos.txt'):
				city_pos=_load_json(path+'city_pos.txt')
				X_pos,Y_pos=[],[]
				for x in city_pos:
					X_pos.append(float(x[0][0])+5) #点的坐标
					Y_pos.append(float(x[0][1])+5)
					if self.draw_region_arg['city_name']:
						plt.text(float(x[1][0])+float(x[1][1])+5,float(x[1][2])+float(x[1][3])+5,x[1][6],fontsize=int(x[1][4])-2,\
							color=self.draw_region_arg['textColor'],ha=self.pos_map[x[1][5]])
				plt.plot(X_pos,Y_pos,'o',color=self.draw_region_arg['point_color'],\
					markersize=self.draw_region_arg['point_size'])

		if title:
			if title is True:
				title=region_name
			plt.title(title)
		plt.gca().set_aspect("equal")
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.gcf().tight_layout()
		if img_path:
			plt.savefig(img_path,facecolor=BC)
			plt.cla()
		else:
			plt.show()

	def Draw_country_county(self,LC='w',FC=[30,120,180],BC='w',lw=0.5,title=None,img_path=None,arg={}):
		'''
			@func: 全国层面绘制县级区划（尚不完善）
		'''
		if not os.path.exists('Data/country_county.txt'):
			print('全国层面的县级区划数据不存在~')
			return
		print('正在解析数据，请稍候~')
		data=_load_json('Data/country_county.txt')

		# 设置绘图选项
		for k,v in arg.items():
			if k in self.country_county_arg:
				self.country_county_arg[k]=v

		# 设置颜色
		LC,FC,BC=self.__set_color([LC,FC,BC])
		plt.figure(facecolor=BC) #背景色

		islands=data['daodian'] #岛屿路径
		del data['daodian']
		for k,v in data.items():
			for pos in v:
				X_pos=pos[0]
				Y_pos=pos[1]
				if LC:
					plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
				if FC:
					plt.fill(X_pos,Y_pos,color=FC)

		# 是否绘制岛屿
		if self.country_county_arg['island']:
			for pos in islands:
				X_pos=pos[0]
				Y_pos=pos[1]
				plt.fill(X_pos,Y_pos,color=self.country_county_arg['island_color'])

		if title:
			plt.title(title)
		plt.gca().set_aspect("equal")
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.gcf().tight_layout()
		if img_path:
			plt.savefig(img_path,facecolor=BC)
			plt.cla()
		else:
			plt.show()

	def Bubble(self,src,by_val=True,color='#F4E925',alpha=0.9,min_s=20,max_s=250,LC='k',FC='#323C48',BC='#505D70',lw=0.5,title=None,img_path=None):
		'''
			@func: 在全国地图上绘制气泡图
			@param:
				src: 数据文件路径(csv)
					2列，第1列为区域编码(默认6位，不足6位右端补0)或区域名称，第2列为相应的数据；
					3列，前两列是坐标的经纬度，第3列为相应的数据
				by_val: Bool
						True: 按值的绝对大小显示
						False: 按值的相对大小显示
				color: 圆圈的颜色
						str: 如'r','b'，或十六进制格式，"#F4E925"
						int: RGB格式，值的范围为0-255
						camp: matplotlib允许的cmap参数，圆的颜色随值的大小变化
						alpha: 圆填充颜色的透明度
						min_s: 圆的最小半径
						max_s: 圆的最大半径
						LC: 地图各区域边的颜色，允许RGB格式、十六进制格式和颜色的字母缩写
						FC: 地图各区域的填充颜色，允许的值同LC
						BC: 地图背景色，允许的值同LC
						lw: 地图各区域边的线宽
						title: 图片的标题，默认无
						img_path: 图片的存储路径(包含图片名)，默认None，即直接显示
			@example:
				D=Draw_Map()
				D.Bubble(src='Example/GDP-2018.csv')
		'''
		if not os.path.exists(src):
			print("数据文件不存在！")
			return
		raw_data=_load_csv(src)
		if not os.path.exists('Data/code_pos.csv'):
			print('区域坐标文件(code_pos.csv)缺失~')
			return

		code_pos=_load_csv('Data/code_pos.csv')  #导入区域坐标文件
		pos={}
		for x in code_pos:
			pos[x[0]]=[float(x[1]),float(x[2])]

		center_pos=[] #在地图中绘制的圆的中心坐标+对应的值
		value=[]  #每个点对应的值
		if len(raw_data[0])==2:
			if raw_data[0][0].isdigit():  #第一列为编码
				for x in raw_data:
					key=x[0].rjust(6,'0')
					if key in pos.keys():
						center_pos.append(pos[key]+[float(x[1])])
						value.append(float(x[1]))
					else:
						print('尚未获取编码%s对应的坐标'%key)
			else: #第一列为区域名称
				if not os.path.exists('Data/code_map.txt'):
					print('区域编码文件(code_map.txt)缺失~')
					return
				code_map=_load_json('Data/code_map.txt')  #导入区域编码文件
				for x in raw_data:
					has_got=False
					for k,v in code_map.items():
						if x[0] in v:
							if k in pos.keys():
								center_pos.append(pos[k]+[float(x[1])])
								value.append(float(x[1]))
							else:
								print('尚未获取编码%s对应的坐标'%k)
							has_got=True
							break
					if not has_got:
						print('未查找到名称为%s对应的区域'%x[0])
		else:
			for x in raw_data:
				coord=self.trans_coord([float(x[0]),float(x[1])])
				center_pos.append(coord+[float(x[2])])
				value.append(float(x[2]))

		# 计算每个坐标对应的圆的半径
		if by_val:
			max_v,min_v=max(value),min(value)
			delta=max_v-min_v
			for i,x in enumerate(center_pos):
				size=min_s+(x[2]-min_v)*(max_s-min_s)/delta
				center_pos[i].append(size)
		else:
			center_pos.sort(key=lambda x:x[2])
			for i,x in enumerate(center_pos):
				size=min_s+(i+1)*(max_s-min_s)/len(center_pos)
				center_pos[i].append(size)

		# 绘制地图背景
		if not os.path.exists('Data/Country'):
			print('省份图形文件不存在~')
			return
		if not os.path.exists('Data/province.txt'):
			print('省份编码文件不存在')
			return
		else:
			province_code=_load_json('Data/province.txt')
		province_shape_file=[x for x in os.listdir('Data/Country/') if len(x)==10 and x[-5]=='0'] #各省图形文件

		# 设置颜色
		LC,FC,BC,color=self.__set_color([LC,FC,BC,color])
		plt.figure(facecolor=BC) #背景色
		from matplotlib.colors import Normalize

		for x in province_shape_file:
			parts=_load_json('Data/Country/'+x)
			for part in parts:
				shape=part[0][1:-1]
				pos=[[float(y) for y in t.split(',')] for t in shape.split('L')]
				X_pos=[t[0] for t in pos]
				Y_pos=[t[1] for t in pos]
				X_pos.append(X_pos[0])
				Y_pos.append(Y_pos[0])
				if LC:
					plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
				if FC:
					plt.fill(X_pos,Y_pos,color=FC)

		# 绘制圆圈
		data={'X':[],'Y':[],'s':[]}
		for val in center_pos:
			data['X'].append(val[0])
			data['Y'].append(val[1])
			data['s'].append(val[3])
		if isinstance(color,str) and len(color)>1 and color[0]!='#':
			plt.scatter('X','Y',s='s',data=data,c='s',alpha=alpha,cmap=color,zorder=2)
		else:
			plt.scatter('X','Y',s='s',data=data,c=color,alpha=alpha,zorder=2)

		# 绘制南海区域
		if not os.path.exists('Data/Country/Sea/sea_path.txt'):
			print('南海区域数据(sea_path.txt)缺失~') #必须绘制南海区域
			return
		else:
			sea_path=_load_json('Data/Country/Sea/sea_path.txt')
			for pos in sea_path:
				X_pos=pos[0]
				Y_pos=pos[1]
				plt.plot(X_pos,Y_pos,color=LC,linewidth=lw)
			plt.plot([740,855,855,740,740],[415,415,570,570,415],color=LC,linewidth=lw*1.2)

		if title:
			plt.title(title)
		plt.gca().set_aspect("equal")
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.gcf().tight_layout()
		if img_path:
			plt.savefig(img_path,facecolor=BC)
			plt.cla()
		else:
			plt.show()

	def __set_color(self,color):
		'''将RGB格式的线条颜色、填充色、背景色从0-255范围内转换到0-1之间'''
		outVal=[]
		for c in color:
			if c:
				if isinstance(c,list) or isinstance(c,tuple):
					outVal.append([x/255 for x in c])
				else:
					outVal.append(c)
		return outVal

	def __get_cmap(self,data,FC,by_val=True):
		'''
		@func: 计算cmap对应的颜色值
		'''
		color_map={}
		cmap=plt.get_cmap(FC)
		if by_val:  #根据值的大小设置颜色
			max_v,min_v=max(data.values()),min(data.values())
			delta=max_v-min_v #值的极差
			for k,v in data.items():
				color_map[k]=cmap((v-min_v-delta*0.05)/(delta*1.1))
		else:  #根据值的排名设置颜色
			temp_val=sorted([(k,v) for k,v in data.items()],key=lambda x:x[1])
			for i,x in enumerate(temp_val):
				color_map[x[0]]=cmap((i+1)/(len(temp_val)+1))
		return color_map

	def del_notes(self,src,word,first=True,remain_file=True):
		'''
			@func:删除地图Notes中不需要的名称(如新疆中的“1”)
			@param:
				src: notes.txt文件所在文件夹的路径
				word: 需要删除的关键词
				first: 是否只删除遇到的第一个词
				remain_file: 是否保留原有的notes文件，若保留，则在原文件名加上_old
			@example:
				D.del_notes('Data/Province/65-新疆','1')
		'''
		if src[-1]!='/': src+='/'
		path=src+'notes.txt'
		if os.path.exists(path):
			notes=_load_json(path)
			for i,x in enumerate(notes):
				note=x[4]
				name=''.join([x[2] for x in note])
				if name==word:
					del notes[i]
					print('已删除'+word)
					if first: break
			if remain_file:
				if os.path.exists(src+'notes_old.txt'):
					os.remove(src+'notes_old.txt')
				os.rename(path,src+'notes_old.txt')
			_dump_json(notes,path)
		else:
			print('notes文件不存在~')

	def trans_coord(self,coord):
		'''
			@func: 将经纬度转换成全国地图中的坐标
			@param:
				coord: list,[x,y], 一个点的经纬度:[经度,维度]
					   list of list,[[x,y],[x,y],...] 一组点的经纬度
			@example:
				D=Draw_Map()
				val=D.trans_coord([79.023836,34.369424])
		'''
		val=coord[0]
		if isinstance(val,float) or isinstance(val,int):
			x=coord[0]*11.67151-750.54
			y=(-0.107987882)*coord[1]**2-6.92431742*coord[1]+706.272492
			return [x,y]
		elif isinstance(val,list):
			outVal=[]
			for v in coord:
				x=v[0]*11.67151-750.54
				y=(-0.107987882)*v[1]**2-6.92431742*v[1]+706.272492
				outVal.append([x,y])
			return outVal

def search_region_code(name):
	'''
		@func:查询区域名称对应的编码，匹配所有含有name的区域
		@param:
			name: str, 区域名称
		@example：
			search_region_code('朝阳')
			>>>北京市朝阳区: 110105
			>>>辽宁省朝阳市: 211300
			>>>辽宁省朝阳市朝阳县: 211321
			>>>吉林省长春市朝阳区: 220104
	'''
	if not os.path.exists('Data/code_map.txt'):
		print('区域编码文件(code_map.txt)不存在')
		return
	all_codes=_load_json('Data/code_map.txt')
	for k,v in all_codes.items():
		if name in v:
			fullName=all_codes[k[:2]+'0000']
			if k[2:4]!='00':
				if k[:2] not in ['11','12','31','50']:
					fullName+=all_codes[k[:4]+'00']
			if k[4:]!='00':
				fullName+=v
			print('%s: %s'%(fullName,k))

def search_code_name(code):
	'''
		@func: 查询code对应的区域名称
		@code:
			str,int:单个区域的编码
			list：一组区域的编码
	'''
	if not os.path.exists('Data/code_map.txt'):
		print('区域编码文件(code_map.txt)不存在')
		return
	all_codes=_load_json('Data/code_map.txt')

	if not isinstance(code,list):
		code=[code]
	for x in code:
		x=str(x).ljust(6,'0')
		if x in all_codes:
			print('%s: %s'%(x,all_codes[x]))
		else:
			print('区域编码%s对应的名称未知'%x)

def _dump_csv(data,path,method='w',coding='utf-8',sep=','):
	try:
		with open(path,method,encoding=coding) as f:
			for x in data:
				line=sep.join(list(map(str,x)))+'\n'
				f.write(line)
	except Exception as e:
		print('Error:%s'%e)

def _load_csv(path,sep=',',coding='utf-8'):
	try:
		with open(path,'r',encoding=coding) as f:
			data=[x.strip().split(sep) for x in f.readlines()]
		return data
	except Exception as e:
		print('Error:%s'%e)

def _dump_json(data,path,method='w',coding='utf-8'):
	try:
		with open(path,method,encoding=coding) as f:
			json.dump(data,f)
	except Exception as e:
		print('Error:%s'%e)

def _load_json(path,coding='utf-8'):
	try:
		with open(path,encoding=coding) as f:
			data=json.load(f)
		return data
	except Exception as e:
		print('Error:%s'%e)

if __name__ == '__main__':
	M=Get_Map()
	D=Draw_Map()