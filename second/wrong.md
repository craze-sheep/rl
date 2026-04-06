![alt text](image.png)

32G显存可是我只有8G(4060)



(rl) PS D:\rl\second> & C:\Users\lzy\anaconda3\envs\rl\python.exe d:/rl/second/train-8G.py
pybullet build time: Oct 21 2025 12:15:52
argv[0]=--background_color_red=0.8745098114013672
argv[1]=--background_color_green=0.21176470816135406
argv[2]=--background_color_blue=0.1764705926179886
🚀 4060 复现版启动 | 任务: PandaPickAndPlace-v3 | 设备: cuda
Ep:     0 | Rew: -50.0 | Suc: 0.00 | Alpha: 0.200 | Steps: 50
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/


文件锁死被打断了
Traceback (most recent call last):
  File "d:\rl\second\train-8G.py", line 270, in <module>
    train()
  File "d:\rl\second\train-8G.py", line 259, in train
    torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_latest.pth"))
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\torch\serialization.py", line 1003, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\torch\serialization.py", line 865, in _open_zipfile_writer
    return container(name_or_buffer)  # type: ignore[arg-type]
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\torch\serialization.py", line 829, in __init__
    torch._C.PyTorchFileWriter(
RuntimeError: [enforce fail at inline_container.cc:745] . open file failed with error code: 1224

画图也把显存画爆了
Ep: 12000 | Rew: -47.0 | Suc: 0.06 | Alpha: 0.006 | Steps: 578192
Traceback (most recent call last):
  File "d:\rl\second\train-8G.py", line 270, in <module>
    train()
  File "d:\rl\second\train-8G.py", line 255, in train
    save_curve(metrics['rewards'], metrics['success'], PLOT_FILE)
  File "d:\rl\second\train-8G.py", line 131, in save_curve
    plt.savefig(path)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\pyplot.py", line 1250, in savefig
    res = fig.savefig(*args, **kwargs)  # type: ignore[func-returns-value]
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\figure.py", line 3490, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backend_bases.py", line 2186, in print_figure
    result = print_method(
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backend_bases.py", line 2042, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backends\backend_agg.py", line 481, in print_png
    self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backends\backend_agg.py", line 429, in _print_pil
    FigureCanvasAgg.draw(self)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backends\backend_agg.py", line 382, in draw
    self.figure.draw(self.renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\artist.py", line 94, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\figure.py", line 3257, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\axes\_base.py", line 3226, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\lines.py", line 821, in draw
    renderer.draw_path(gc, tpath, affine.frozen())
  File "C:\Users\lzy\anaconda3\envs\rl\lib\site-packages\matplotlib\backends\backend_agg.py", line 130, in draw_path
    self._renderer.draw_path(gc, path, transform, rgbFace)
MemoryError: bad allocation




PS D:\rl\second> (C:\Users\lzy\anaconda3\Scripts\activate) ; (conda activate rl)
(rl) PS D:\rl\second> & C:\Users\lzy\anaconda3\envs\rl\python.exe d:/rl/second/train-8G.py
pybullet build time: Oct 21 2025 12:15:52
argv[0]=--background_color_red=0.8745098114013672
argv[1]=--background_color_green=0.21176470816135406
argv[2]=--background_color_blue=0.1764705926179886
Unhandled exception caught in c10/util/AbortHandler.h
00007FFDC72C901600007FFDC72BF3B0 torch_python.dll!torch::autograd::THPCppFunction_requires_grad [<unknown file> @ <unknown line number>]
00007FFFA6BE19D700007FFFA6BE19C0 ucrtbase.dll!terminate [<unknown file> @ <unknown line number>]
00007FF799AF1C1600007FF799AF1110 python.exe!OPENSSL_Applink [<unknown file> @ <unknown line number>]
00007FFFA75F570300007FFFA75F5510 KERNELBASE.dll!UnhandledExceptionFilter [<unknown file> @ <unknown line number>]
00007FFFA9CAA4D300007FFFA9CA7E50 ntdll.dll!strncpy [<unknown file> @ <unknown line number>]
00007FFFA9C618D300007FFFA9C61840 ntdll.dll!_C_specific_handler [<unknown file> @ <unknown line number>]
00007FFFA9CA61CF00007FFFA9CA6130 ntdll.dll!_chkstk [<unknown file> @ <unknown line number>]
00007FFFA9B523A700007FFFA9B51E10 ntdll.dll!RtlLocateExtendedFeature [<unknown file> @ <unknown line number>]
00007FFFA9B4A96100007FFFA9B4A740 ntdll.dll!RtlRaiseException [<unknown file> @ <unknown line number>]
00007FFFA75A73FA00007FFFA75A7370 KERNELBASE.dll!RaiseException [<unknown file> @ <unknown line number>]
00007FFF5FCD52C700007FFF5FCD5230 VCRUNTIME140.dll!CxxThrowException [<unknown file> @ <unknown line number>]
00007FFD95683ECF00007FFD9567DE70 pybullet.cp310-win_amd64.pyd!b3ConnectPhysicsDirect [<unknown file> @ <unknown line number>]
00007FFD958B2B7900007FFD9577D170 pybullet.cp310-win_amd64.pyd!PyInit_pybullet [<unknown file> @ <unknown line number>]
00007FFD95675AD200007FFD956755B0 pybullet.cp310-win_amd64.pyd!b3ConnectPhysicsUDP [<unknown file> @ <unknown line number>]
00007FFD956F06D200007FFD956EFAE0 pybullet.cp310-win_amd64.pyd!b3CreateInProcessGraphicsServerAndConnectMainThreadSharedMemory [<unknown file> @ <unknown line number>]
00007FFD9569547000007FFD9567DE70 pybullet.cp310-win_amd64.pyd!b3ConnectPhysicsDirect [<unknown file> @ <unknown line number>]
00007FFD9567DEAA00007FFD9567DE70 pybullet.cp310-win_amd64.pyd!b3ConnectPhysicsDirect [<unknown file> @ <unknown line number>]
00007FFD95760A9D00007FFD9571E450 pybullet.cp310-win_amd64.pyd!executePluginCommand_tinyRendererPlugin [<unknown file> @ <unknown line number>]
00007FFE4A192DAD00007FFE4A191FF0 python310.dll!PyCFunction_GetFlags [<unknown file> @ <unknown line number>]
00007FFE4A14D92800007FFE4A14D7F0 python310.dll!PyObject_MakeTpCall [<unknown file> @ <unknown line number>]
00007FFE4A2576D200007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A253CFE00007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A14DCDE00007FFE4A14DCA0 python310.dll!PyFunction_Vectorcall [<unknown file> @ <unknown line number>]
00007FFE4A14D71100007FFE4A14D660 python310.dll!PyObject_FastCallDictTstate [<unknown file> @ <unknown line number>]
00007FFE4A14DF5500007FFE4A14DEB0 python310.dll!PyObject_Call_Prepend [<unknown file> @ <unknown line number>]
00007FFE4A1BD7A800007FFE4A1B9580 python310.dll!PyType_Ready [<unknown file> @ <unknown line number>]
00007FFE4A1B121500007FFE4A1AFBF0 python310.dll!PyType_Name [<unknown file> @ <unknown line number>]
00007FFE4A14D92800007FFE4A14D7F0 python310.dll!PyObject_MakeTpCall [<unknown file> @ <unknown line number>]
00007FFE4A2576D200007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A253CFE00007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A14DCDE00007FFE4A14DCA0 python310.dll!PyFunction_Vectorcall [<unknown file> @ <unknown line number>]
00007FFE4A14D71100007FFE4A14D660 python310.dll!PyObject_FastCallDictTstate [<unknown file> @ <unknown line number>]
00007FFE4A14DF5500007FFE4A14DEB0 python310.dll!PyObject_Call_Prepend [<unknown file> @ <unknown line number>]
00007FFE4A1BD7A800007FFE4A1B9580 python310.dll!PyType_Ready [<unknown file> @ <unknown line number>]
00007FFE4A1B121500007FFE4A1AFBF0 python310.dll!PyType_Name [<unknown file> @ <unknown line number>]
00007FFE4A14D92800007FFE4A14D7F0 python310.dll!PyObject_MakeTpCall [<unknown file> @ <unknown line number>]
00007FFE4A2576D200007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A253CFE00007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A14DCDE00007FFE4A14DCA0 python310.dll!PyFunction_Vectorcall [<unknown file> @ <unknown line number>]
00007FFE4A14D71100007FFE4A14D660 python310.dll!PyObject_FastCallDictTstate [<unknown file> @ <unknown line number>]
00007FFE4A14DF5500007FFE4A14DEB0 python310.dll!PyObject_Call_Prepend [<unknown file> @ <unknown line number>]
00007FFE4A1BD7A800007FFE4A1B9580 python310.dll!PyType_Ready [<unknown file> @ <unknown line number>]
00007FFE4A1B121500007FFE4A1AFBF0 python310.dll!PyType_Name [<unknown file> @ <unknown line number>]
00007FFE4A14DBD700007FFE4A14DAE0 python310.dll!PyObject_Call [<unknown file> @ <unknown line number>]
00007FFE4A2578CB00007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A2529BA00007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A14DCDE00007FFE4A14DCA0 python310.dll!PyFunction_Vectorcall [<unknown file> @ <unknown line number>]
00007FFE4A24EB7100007FFE4A24EA20 python310.dll!PyOS_URandomNonblock [<unknown file> @ <unknown line number>]
00007FFE4A2576D200007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A253C8600007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A14DCDE00007FFE4A14DCA0 python310.dll!PyFunction_Vectorcall [<unknown file> @ <unknown line number>]
00007FFE4A24EB7100007FFE4A24EA20 python310.dll!PyOS_URandomNonblock [<unknown file> @ <unknown line number>]
00007FFE4A2576D200007FFE4A2572E0 python310.dll!PyEval_GetFuncDesc [<unknown file> @ <unknown line number>]
00007FFE4A25367000007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A255DD400007FFE4A2506E0 python310.dll!PyEval_EvalFrameDefault [<unknown file> @ <unknown line number>]
00007FFE4A2C904900007FFE4A2C8D70 python310.dll!PyRun_FileExFlags [<unknown file> @ <unknown line number>]
00007FFE4A2C911800007FFE4A2C8D70 python310.dll!PyRun_FileExFlags [<unknown file> @ <unknown line number>]
00007FFE4A2C8D3800007FFE4A2C8BD0 python310.dll!PyRun_StringFlags [<unknown file> @ <unknown line number>]
00007FFE4A2C6B2E00007FFE4A2C6880 python310.dll!PyRun_SimpleFileObject [<unknown file> @ <unknown line number>]