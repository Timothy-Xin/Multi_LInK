import os
import uuid
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--port", type=int, default=1238, help="Port number for the local server")
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument("--static_folder", type=str, default='static', help="Folder to store static files")
argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/',
                       help='The folder to store the checkpoint')
argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
argparser.add_argument('--embedding_folder', type=str, default='./Embeddings/',
                       help='The folder to store the embeddings')
args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import gradio as gr
from LInK.demo import draw_html, draw_script, css
from LInK.Solver import solve_rev_vectorized_batch_CPU
from LInK.CAD import get_layers, create_3d_html
from LInK.OptimJax import PathSynthesis
from LInK.OptimJax import preprocess_curves
from LInK.OptimJax import preprocess_multi_curves_as_whole
from pathlib import Path
import jax

import numpy as np
import pickle
import torch
import json

# turn off gradient computation
torch.set_grad_enabled(False)

# check if the static folder exists
if not Path(args.static_folder).exists():
    os.mkdir(args.static_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
if not os.path.exists(args.checkpoint_folder) or not os.path.exists(
        os.path.join(args.checkpoint_folder, args.checkpoint_name)):
    raise ValueError(
        'The checkpoint file does not exist please run Download.py to download the checkpoints or provide the correct path.')

# load the model
if device == 'cpu':
    with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
        Trainer = pickle.load(f)
else:
    with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
        Trainer = pickle.load(f)

Trainer.model_base = Trainer.model_base.to('cpu')
Trainer.model_mechanism = Trainer.model_mechanism.to('cpu')
Trainer.model_input.compile()

# load data
if not os.path.exists(args.data_folder) or not os.path.exists(
        os.path.join(args.data_folder, 'target_curves.npy')) or not os.path.exists(
    os.path.join(args.data_folder, 'connectivity.npy')) or not os.path.exists(
    os.path.join(args.data_folder, 'x0.npy')) or not os.path.exists(
    os.path.join(args.data_folder, 'node_types.npy')):
    raise ValueError(
        'All or some of the data does not exist please run Download.py to download the data or provide the correct path.')

if not os.path.exists(args.embedding_folder) or not os.path.exists(
        os.path.join(args.embedding_folder, 'embeddings.npy')):
    raise ValueError(
        'The embedding file does not exist please run Download.py to download the embedding file or run Precompute.py to recompute them or provide the correct path.')

emb = np.load(os.path.join(args.embedding_folder, 'embeddings_subset.npy'))
# emb = torch.tensor(emb).float().to(device)
emb = jax.numpy.array(emb, dtype=jax.numpy.float32)
As = np.load(os.path.join(args.data_folder, 'connectivity_subset.npy'))
x0s = np.load(os.path.join(args.data_folder, 'x0_subset.npy'))
node_types = np.load(os.path.join(args.data_folder, 'node_types_subset.npy'))
curves = np.load(os.path.join(args.data_folder, 'target_curves_subset.npy'))
sizes = (As.sum(-1) > 0).sum(-1)
alpha = np.load('./TestData/alphabet.npy', allow_pickle=True)

# 通过曲线起点到第一个点与最后一个点的距离判断是否为部分曲线
partials = []
for i in range(len(alpha)):
    curve = preprocess_curves(alpha[i:i + 1])[0]
    partial = True
    t = np.linalg.norm(curve[0] - curve[1])
    e = np.linalg.norm(curve[0] - curve[-1])
    if e / t <= 1.1:
        partial = False
    partials.append(partial)

    # rotate curve 180 degrees
    R = np.array([[-1, 0], [0, 1]]) @ np.array([[-1, 0], [0, -1]])
    alpha[i] = np.dot(R, alpha[i].T).T

# 字母搞反了，调换一下
a = np.copy(alpha[15])
alpha[15] = alpha[16]
alpha[16] = a

torch.cuda.empty_cache()


# 初始化机制合成器
# n_freq：用于曲线平滑处理的频率数量。
# maximum_joint_count：允许的连杆机制的最大关节数量。
# time_steps：模拟轨迹时的时间步数。
# top_n：从嵌入空间中筛选出的最相似的候选机制数量。
# init_optim_iters：对所有候选机制进行初步优化的迭代次数。
# top_n_level2：从初步优化后的候选机制中筛选出的最终优化数量。
# BFGS_max_iter：使用 BFGS 算法进行最终优化的最大迭代次数。
def create_synthesizer(n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter):
    # mask = (sizes<=maximum_joint_count)
    synthesizer = PathSynthesis(Trainer, curves, As, x0s, node_types, emb, BFGS_max_iter=BFGS_max_iter, n_freq=n_freq,
                                optim_timesteps=time_steps, top_n=top_n, init_optim_iters=init_optim_iters,
                                top_n_level2=top_n_level2)
    return synthesizer


# 根据合成的连杆机制信息生成一个3D模型，并将其保存为HTML文件以便在Gradio界面上显示
def make_cad(synth_out, partial, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Generating 3D Model ...")

    f_name = str(uuid.uuid4())

    A_M, x0_M, node_types_M, start_theta_M, end_theta_M, tr_M = synth_out[0]

    # 计算连杆在不同角度下的运动轨迹
    sol_m = solve_rev_vectorized_batch_CPU(A_M[np.newaxis], x0_M[np.newaxis], node_types_M[np.newaxis],
                                           np.linspace(start_theta_M, end_theta_M, 200))[0]

    # 获取层次结构
    z, status = get_layers(A_M, x0_M, node_types_M, sol_m)

    if partial:
        sol_m = np.concatenate([sol_m, sol_m[:, ::-1, :]], axis=1)

    create_3d_html(A_M, x0_M, node_types_M, z, sol_m, template_path=f'./{args.static_folder}/animation.html',
                   save_path=f'./{args.static_folder}/{f_name}.html')

    return gr.HTML(f'<iframe width="100%" height="800px" src="file={args.static_folder}/{f_name}.html"></iframe>',
                   label="3D Plot", elem_classes="plot3d")


gr.set_static_paths(paths=[Path(f'./{args.static_folder}')])

with gr.Blocks(css=css, js=draw_script) as block:
    syth = gr.State()
    state = gr.State()
    dictS = gr.State()

    with gr.Row():
        intro = gr.Markdown('''
        # MTLS（Multi-trajectory Linkage Path Synthesis）: Learning Joint Representations of Design and Performance Spaces through Contrastive Learning for Mechanism Synthesis.
        MTLS is a novel framework that integrates contrastive learning of performance and design space with optimization techniques for solving complex inverse problems in engineering design with discrete and continuous variables. We focus on the path synthesis problem for planar linkage mechanisms in this application.<br>
        Below, you can draw or upload one or multiple curves and synthesize a mechanism that can trace the curves. You can also adjust the algorithm parameters to see how it affects the solution.
        ''', elem_classes="intro")

    with gr.Row():

        with gr.Column(min_width=350, scale=2):
            canvas = gr.HTML(draw_html)

            # 选择曲线来源（上传文件或预定义曲线）
            curve_source = gr.Radio(
                choices=["Upload Single Curve File", "Choose Predefined Curve", "Upload Multi-Curve File(n<=3)"],
                label="Select Curve Source",
                type="index"
            )

            # 单轨迹文件上传组件
            upload_single_file = gr.File(label="Upload Single Curve File (npy)", visible=False)

            # add predefiened curve choices of alphabet
            curve_choices = gr.Radio(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                 "U", "V", "W", "X", "Y", "Z"], label="Predefined Curves", elem_classes="curve_choices", type='index',
                visible=False)

            btn_submit = gr.Button("Perform Path Synthesis", variant='primary', elem_classes="clr_btn")

            # 多轨迹文件上传组件
            upload_multi_file = gr.File(label="Upload Multi-Curve File (npy)", visible=False)

            btn_multi_submit = gr.Button("Perform Multi-Path Synthesis", variant='primary', elem_classes="clr_btn",
                                         visible=False)

            clr_btn = gr.Button("Clear", elem_classes="clr_btn")

            # checkboxc
            partial = gr.Checkbox(label="Partial Curve", value=False, elem_id="partial")

        with gr.Column(min_width=250, scale=1, visible=True):
            gr.HTML("<h2>Algorithm Parameters</h2>")

            n_freq = gr.Slider(minimum=3, maximum=50, value=7, step=1, label="Number of Frequenceies For smoothing",
                               interactive=True)
            maximum_joint_count = gr.Slider(minimum=6, maximum=20, value=14, step=1, label="Maximum Joint Count",
                                            interactive=True)
            time_steps = gr.Slider(minimum=1000, maximum=3000, value=2000, step=500,
                                   label="Number of Simulation Time Steps", interactive=True, visible=False)
            top_n = gr.Slider(minimum=50, maximum=1000, value=300, step=50, label="Top N Candidates To Start With",
                              interactive=True, visible=False)
            init_optim_iters = gr.Slider(minimum=10, maximum=50, value=20, step=10,
                                         label="Initial Optimization Iterations On All Candidates", interactive=True)
            top_n_level2 = gr.Slider(minimum=10, maximum=100, value=40, step=10,
                                     label="Top N Candidates For Final Optimization", interactive=True, visible=False)
            BFGS_max_iter = gr.Slider(minimum=50, maximum=1000, value=200, step=50,
                                      label="Iterations For Final Optimization", interactive=True)

            storage = gr.HTML('<textarea id="json_text" style="display:none;"></textarea>')

    with gr.Row():
        with gr.Row():
            with gr.Column(min_width=250, scale=1, visible=True):
                gr.HTML('<h2>Algorithm Outputs</h2>')
                progl = gr.Label({"Progress": 0}, elem_classes="prog", num_top_classes=1)

    with gr.Row():
        with gr.Column(min_width=250, visible=True):
            og_plt = gr.Plot(label="Original Input", elem_classes="plotpad")
        with gr.Column(min_width=250, visible=True):
            smooth_plt = gr.Plot(label="Smoothed Drawing", elem_classes="plotpad")

    with gr.Row():
        candidate_plt = gr.Plot(label="Initial Candidates", elem_classes="plotpad")

    with gr.Row():
        mechanism_plot = gr.Plot(label="Solution", elem_classes="plotpad")

    with gr.Row():
        plot_3d = gr.HTML('<iframe width="100%" height="800px" src="file=static/filler.html"></iframe>',
                          label="3D Plot", elem_classes="plot3d")


    # 处理单轨迹文件上传
    def handle_single_upload(file):
        try:
            # 读取上传的 .npy 文件
            curve_data = np.load(file.name)
            # 解析并绘制曲线
            curve_upload = preprocess_curves(curve_data[0:1])[0]  # 直接处理整个 curve_data，因为只有一条曲线
            partial_upload = True
            t = np.linalg.norm(curve_upload[0] - curve_upload[1])
            e = np.linalg.norm(curve_upload[0] - curve_upload[-1])
            if e / t <= 1.1:
                partial_upload = False
            partials_upload = [partial_upload]

            # rotate curve 180 degrees
            R = np.array([[-1, 0], [0, 1]]) @ np.array([[-1, 0], [0, -1]])
            curve_data[0] = np.dot(R, curve_data[0].T).T

            curve_data = str((80 * (curve_data[0][None])[0] + 350 // 2).tolist())

            return partials_upload, curve_data
        except Exception as e:
            raise ValueError(f"Failed to load .npy file: {e}")


    # 处理多轨迹文件上传
    def handle_multi_upload(file):
        try:
            curve_data = np.load(file.name)  # shape (n_curves, 200, 3)
            coords = curve_data[..., :2]  # (n, 200, 2)
            ids = curve_data[..., 2]  # (n, 200)
            ids_per_curve = ids[:, 0]  # 每条曲线的 id（假设每条曲线的所有点 id 相同）

            # === 统一处理所有曲线，保持相对位置 ===
            processed_coords = preprocess_multi_curves_as_whole(coords)

            # === 分离 id=1 的曲线（应该只有一条） ===
            curve_id_1 = processed_coords[ids_per_curve == 1][0]

            # === 判断是否为 partial curve ===
            t = np.linalg.norm(curve_id_1[0] - curve_id_1[1])
            e = np.linalg.norm(curve_id_1[0] - curve_id_1[-1])
            partial_curve = e / t > 1.1
            partials_upload = [partial_curve]

            # === 坐标变换函数 ===
            def convert_to_str_format(curve):
                transformed = np.dot(np.array([[1, 0], [0, -1]]), curve.T).T
                return str((80 * transformed + 350 // 2).tolist())

            # === id=1 的曲线字符串表示 ===
            curve_id_1_str = convert_to_str_format(curve_id_1)

            # === 提取 id=0 的曲线集合 ===
            curves_id_0 = processed_coords[ids_per_curve == 0]

            # === 分别转换 id=0 的曲线字符串 ===
            curve_id_0_strs = [convert_to_str_format(c) for c in curves_id_0]

            # === 将所有曲线都打包成字符串列表 ===
            all_curve_strs = [curve_id_1_str] + curve_id_0_strs

            return partials_upload, json.dumps([json.loads(s) for s in all_curve_strs])

        except Exception as e:
            raise ValueError(f"Failed to load multi-curve .npy file: {e}")


    # 切换曲线来源的显示
    def toggle_curve_source(choice):
        if choice == 0:  # 上传单轨迹文件
            return (
                gr.update(visible=True),  # upload_single_file
                gr.update(visible=False),  # curve_choices
                gr.update(visible=False),  # upload_multi_file
                gr.update(visible=True),  # btn_submit
                gr.update(visible=False)  # btn_multi_submit
            )
        elif choice == 1:  # 预定义曲线
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            )
        else:  # choice == 2 上传多轨迹文件
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
            )


    curve_source.change(
        toggle_curve_source,
        inputs=[curve_source],
        outputs=[upload_single_file, curve_choices, upload_multi_file, btn_submit, btn_multi_submit]
    )

    # 当用户点击“Perform Path Synthesis”按钮时，禁用一些交互组件以防止重复提交。
    event1 = btn_submit.click(lambda: [None] * 4 + [gr.update(interactive=False)] * 8,
                              outputs=[candidate_plt, mechanism_plot, og_plt, smooth_plt, btn_submit, n_freq,
                                       maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2,
                                       BFGS_max_iter], concurrency_limit=10)

    # 在event1完成后，调用create_synthesizer函数初始化PathSynthesis对象。
    event2 = event1.then(create_synthesizer,
                         inputs=[n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2,
                                 BFGS_max_iter], outputs=[syth], concurrency_limit=10)
    # 在event2完成后，调用demo_sythesize_step_1方法进行第一步合成。
    event3 = event2.then(
        lambda s, x, p: s.demo_sythesize_step_1(np.array([eval(i) for i in x.split(',')]).reshape([-1, 2]) * [[1, -1]],
                                                partial=p), inputs=[syth, canvas, partial],
        js="(s,x,p) => [s,path.toString(),p]", outputs=[state, og_plt, smooth_plt], concurrency_limit=10)

    # 在event3完成后，调用demo_sythesize_step_2方法进行第二步合成。
    event4 = event3.then(lambda sy, s, mj: sy.demo_sythesize_step_2(s, max_size=mj),
                         inputs=[syth, state, maximum_joint_count], outputs=[state, candidate_plt],
                         concurrency_limit=10)

    # 在event4完成后，调用demo_sythesize_step_3方法进行第三步合成。
    event5 = event4.then(lambda sy, s: sy.demo_sythesize_step_3(s, progress=gr.Progress()), inputs=[syth, state],
                         outputs=[mechanism_plot, state, progl], concurrency_limit=10)

    # 在event5完成后，调用make_cad函数生成3D模型。
    event6 = event5.then(make_cad, inputs=[state, partial], outputs=[plot_3d], concurrency_limit=10)

    # 在event6完成后，恢复之前禁用的交互组件。
    event7 = event6.then(lambda: [gr.update(interactive=True)] * 8,
                         outputs=[btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters,
                                  top_n_level2, BFGS_max_iter], concurrency_limit=10)

    # 当用户点击“Perform Multi-Path Synthesis”按钮时
    multi_event1 = btn_multi_submit.click(lambda: [None] * 4 + [gr.update(interactive=False)] * 8,
                                          outputs=[candidate_plt, mechanism_plot, og_plt, smooth_plt, btn_multi_submit,
                                                   n_freq,
                                                   maximum_joint_count, time_steps, top_n, init_optim_iters,
                                                   top_n_level2,
                                                   BFGS_max_iter], concurrency_limit=10)
    multi_event2 = multi_event1.then(create_synthesizer,
                                     inputs=[n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters,
                                             top_n_level2,
                                             BFGS_max_iter], outputs=[syth], concurrency_limit=10)
    multi_event3 = multi_event2.then(
        lambda s, x, p: s.demo_multi_sythesize_step_1(
            np.array([eval(i) for i in x.split(',')]).reshape([-1, 200, 2]) * [1, -1],
            partial=p), inputs=[syth, canvas, partial],
        js="""
                (s, x, p) => {
                    const flat = path.toString();  // 转换成字符串
                    return [s, flat, p];  // 返回后续的值
                }
            """,
        outputs=[state, og_plt, smooth_plt], concurrency_limit=10)


    # multi_event4 = multi_event3.then(lambda sy, s, mj: sy.demo_sythesize_step_2(s, max_size=mj),
    #                                  inputs=[syth, state, maximum_joint_count], outputs=[state, candidate_plt],
    #                                  concurrency_limit=10)
    # multi_event5 = multi_event4.then(lambda sy, s: sy.demo_sythesize_step_3(s, progress=gr.Progress()),
    #                                  inputs=[syth, state],
    #                                  outputs=[mechanism_plot, state, progl], concurrency_limit=10)
    # multi_event6 = multi_event5.then(make_cad, inputs=[state, partial], outputs=[plot_3d], concurrency_limit=10)
    # multi_event7 = multi_event6.then(lambda: [gr.update(interactive=True)] * 8,
    #                                  outputs=[btn_multi_submit, n_freq, maximum_joint_count, time_steps, top_n,
    #                                           init_optim_iters,
    #                                           top_n_level2, BFGS_max_iter], concurrency_limit=10)

    # 将状态值传递给JavaScript函数。
    def aux(state):
        # Pass the state value to the JS function
        return gr.HTML(f'<textarea id="json_text" style="display:none;">{state}</textarea>')


    # 绑定单轨迹文件上传事件
    e1_single_upload = upload_single_file.change(
        handle_single_upload,
        inputs=[upload_single_file],
        outputs=[partial, dictS]
    )
    e2_single_upload = e1_single_upload.then(aux, inputs=[dictS], outputs=[storage])
    e3_single_upload = e2_single_upload.then(None,
                                             js='pre_defined_curve(document.getElementById("json_text").innerHTML)')

    # 绑定多轨迹文件上传事件
    e1_multi_upload = upload_multi_file.change(
        handle_multi_upload,
        inputs=[upload_multi_file],
        outputs=[partial, dictS]
    )
    e2_multi_upload = e1_multi_upload.then(aux, inputs=[dictS], outputs=[storage])
    e3_multi_upload = e2_multi_upload.then(None,
                                           js='pre_multi_defined_curve(document.getElementById("json_text").innerHTML)')

    # 绑定选择轨迹事件
    e1 = curve_choices.change(lambda idx: (partials[idx], str((80 * (alpha[idx][None])[0] + 350 // 2).tolist())),
                              outputs=[partial, dictS], inputs=[curve_choices])
    e2 = e1.then(aux, inputs=[dictS], outputs=[storage])
    e3 = e2.then(None, js='pre_defined_curve(document.getElementById("json_text").innerHTML)')

    block.load()
    # 画布清除事件处理
    clr_btn.click(lambda x: x, js='document.getElementById("sketch").innerHTML = ""')

block.launch(root_path='/linkage', server_name='localhost', server_port=args.port, share=True, max_threads=200,
             inline=True)
