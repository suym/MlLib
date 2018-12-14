# -*- mode: python -*-

tensorflow_location = '/root/suym/local/anaconda2/lib/python2.7/site-packages/tensorflow'
model_name = 'mlModel'
add_model_pathex = []

def add_tensorflow_so():
    '''
    需要修改*/site-packages/tensorflow/python/framework/load_library.py文件，在load_op_library(library_filename)函数
    中加入library_filename = library_filename.split('/')[-1]。注意在pyinstaller打包好后记得修改回来
    '''
    import os
    tensorflow_binaries = []
    for dir_name, sub_dir_list, fileList in os.walk(tensorflow_location):
        for file in fileList:
            if file.endswith(".so"):
                full_file = dir_name + '/' + file
                tensorflow_binaries.append((full_file, '.'))
    return tensorflow_binaries 
def add_pygal_data():
    from PyInstaller.utils.hooks import collect_data_files
    pygal_datas = collect_data_files('pygal')
    return pygal_datas

block_cipher = None

a = Analysis(['%s.py'%model_name],
             pathex=add_model_pathex,
             binaries=add_tensorflow_so(),
             datas=add_pygal_data(),
             hiddenimports=['_sysconfigdata','cython','sklearn','sklearn.ensemble','sklearn.neighbors.typedefs','sklearn.neighbors.quad_tree','sklearn.tree._utils','scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name=model_name,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name=model_name)