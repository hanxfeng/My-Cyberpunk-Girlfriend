这里是用于服务器/电脑的后端代码  
使用方法及注意事项：  
1.使用python3.10版本，高版本可能对faiss不兼容  
2.在flaks文件夹内打开终端然后输入python api.py即可  
需要首先在faiss_train.py中根据自己的数据创建faiss索引，然后才能在api与model_function中使用，faiss索引默认保存在templates文件夹中  
下载到本地的模型放在models文件夹中  
数据文件默认从templates文件夹中读取，为json文件，其他格式需自行修改代码  
