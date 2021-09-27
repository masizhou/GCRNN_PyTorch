import Map_of_China.Map_of_China as Map


# 创建获取地图数据对象
M = Map.Get_Map()
# 创建绘制地图对象
D = Map.Draw_Map()


# #获取首页全国各省的地图数据
M.Get_country()
# 江苏地图
D.Draw_single_shape('Data/Country/320000.txt', LC=[116, 169, 221],
                                   FC=[96, 139, 219], BC='w', lw=2, title='江苏轮廓',
                                   img_path='jiansu.jpg')


# 获取各省份内各地级行政区划的数据
M.Get_provinces([32])
# 南京地图
D.Draw_single_shape('Data/Province/32-江苏/320100.txt', LC='k', FC=None,
                                   title='南京轮廓', img_path='nanjing.jpg')