import numpy as np
import torch
from rd_generator import RDGenerator
from tools import GSParams, init_state_batch
import matplotlib.animation as animation
from animation import get_initial_artists, updatefig
from power_spectrum_2d import windowed_ps_2d_loss, ps_2d_loss, VGGGramLoss

import matplotlib.pyplot as plt

import sys
import copy


def print_usage_and_exit():
    print(
        "Please specify the task you want to perform:\n"
        "1. generate targets\n"
        "2. test training forward (starting from noisy random parameters)\n"
        "3. load existing targets and show the first 4 of them\n"
        "4. show stepping animation\n"
        "5. plot loss values on a slice of the parameter space\n"
        "Type: python pattern_gen_outside_training.py <number of your option> <optional: path to existing targets>"
    )
    sys.exit(1)


# ---- parse command line ----
if len(sys.argv) < 2:
    print_usage_and_exit()

try:
    task_id = int(sys.argv[1])
except ValueError:
    print_usage_and_exit()

if task_id not in (1, 2, 3, 4, 5):
    print_usage_and_exit()

if task_id == 3:
    target_pt_path = sys.argv[2]
task = 'test training forward' if task_id == 2 else \
       'generate targets' if task_id == 1 else \
       'show stored targets' if task_id ==3 else \
       'animation' if task_id == 4 else \
       'landscape' if task_id == 5 else \
       'error'
print(f"Current task: {task}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)
    # To produce slope compare plots, set the next line to True and put the pair of losses in compare_pair.
    save_animation = False
    three_d = True
    vgg = VGGGramLoss(device)
    candidate_losses = {
        'non-windowed': lambda x1, x2: ps_2d_loss(x1, x2)[3].item(),
        'windowed': lambda x1, x2: windowed_ps_2d_loss(x1, x2)[3].item(),
        'vgg': lambda x1, x2: vgg(x1, x2).item(),
    }
    params = [GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.068),
              GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.068)]
    # params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)

    print('Generating initial batch...')
    u, v = init_state_batch(B=4, H=128, W=128, device=device)
    if task == 'generate targets':
        v_batch = RDGenerator.generate_gray_scott_target_batch(u, v, params=params[3], device=device, tol=1e-8,
                                                               max_steps=50000)
    elif task == 'test training forward':
        gen = RDGenerator()
        v_batch, overflow = gen.simulate_to_steady_trunc_bptt(u, v, device=device)[1:3]
        if overflow:
            print("The output images may not be accurate. Values out of [0, 1] range during stepping.")
    elif task == 'show stored targets':
        v_batch = torch.load(target_pt_path, map_location=device)
    elif task == 'animation':
        print("Modify the bool `save_animation` to True if you want to save it as a file instead of show it directly.")
        if type(params) is list:
            l = len(params)
            repeat = [(l-1) // 4 + 1, 1, 1, 1]
            u_animation = [u_item.squeeze(0) for u_item in u.repeat(repeat)[:l]]
            v_animation = [v_item.squeeze(0) for v_item in v.repeat(repeat)[:l]]
        else:
            u_animation, v_animation = u[0].squeeze(0), v[0].squeeze(0)

        fig, im, txt = get_initial_artists(v_animation, per_line=7)
        updates_per_frame = 5
        dt = 1.0
        animation_arguments = (updates_per_frame, im, txt, u_animation, v_animation, params, dt)
        ani = animation.FuncAnimation(fig,  # matplotlib figure
                                      updatefig,  # function that takes care of the update
                                      fargs=animation_arguments,  # arguments to pass to this function
                                      interval=1,  # update every `interval` milliseconds
                                      frames=10000,
                                      blit=False,  # optimize the drawing update
                                      )
        if save_animation:
            print("Saving animation...")
            ani.save(
                "gray_scott.mp4",
                writer="ffmpeg",
                fps=30,
                dpi=150
            )
            print("Saved to gray_scott.mp4")
        # show the animation
        plt.show()
        exit(0)
    elif task == 'landscape':
        if len(sys.argv) < 6 and sys.argv[2] != "param_in_file":
            print("Type `python pattern_gen_outside_training.py 5 <OPTIONAL: loss function name> "
                  "<name_of_param_to_slice> <start_value> <end_value> <slice_num>` or "
                  "`python pattern_gen_outside_training.py 5 <OPTIONAL: loss function name> "
                  "<name_of_param_to_slice_1> <start_value_1> <end_value_1> <slice_num_1> "
                  "<name_of_param_to_slice_2> <start_value_2> <end_value_2> <slice_num_2>` or "
                  "`python pattern_gen_outside_training.py 5 <OPTIONAL: loss function name> param_in_file`.")
            print("Modify the bool `three_d` to True to turn on 3d plot with two parameters; "
                  "when use the `param_in_file` command line option, modify the bool `compare_loss_funcs` to True "
                  "to compare a pair of loss functions (defined in `compare_pair`) "
                  "on a list of parameter sets defined in the Python file.")
            sys.exit(1)
        p = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)
        up, vp = u.clone(), v.clone()
        # v_batch = RDGenerator.generate_gray_scott_target_batch(up, vp, params=p, device=device, tol=1e-8,
        #                                                        max_steps=50000)
        _, v_batch, _ = RDGenerator.simulate_constant_steps(up, vp, p, num_steps=40000)
        loss_values = []
        import json, os
        path = "loss_matrix.jsonl"
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    loss_values.append(json.loads(line.strip()))
                f.close()
        loss_function_name = 'non-windowed'
        skip_loss_name_base_index = 0
        if sys.argv[2] in {'non-windowed', 'windowed', 'vgg'}:
            loss_function_name = sys.argv[2]
            skip_loss_name_base_index = 1
        if sys.argv[2 + skip_loss_name_base_index] != "param_in_file":
            if len(sys.argv) > (6 + skip_loss_name_base_index):
                vname, vmin, vmax, steps = (sys.argv[2 + skip_loss_name_base_index],
                                            float(sys.argv[3 + skip_loss_name_base_index]),
                                            float(sys.argv[4 + skip_loss_name_base_index]),
                                            int(sys.argv[5 + skip_loss_name_base_index]))
                v2name, v2min, v2max, steps2 = (sys.argv[6 + skip_loss_name_base_index],
                                                float(sys.argv[7 + skip_loss_name_base_index]),
                                                float(sys.argv[8 + skip_loss_name_base_index]),
                                                int(sys.argv[9 + skip_loss_name_base_index]))
                values = [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps-1, -1, -1)]
                values2 = [v2min + i * (v2max - v2min) / (steps2 - 1) for i in range(steps2)]
                params = []
                for value in values:
                    p_inner_list = []
                    for value2 in values2:
                        _p = copy.deepcopy(p)
                        setattr(_p, vname, value)
                        setattr(_p, v2name, value2)
                        p_inner_list.append(_p)
                    params.append(p_inner_list)
            else:
                vname, vmin, vmax, steps = (sys.argv[2 + skip_loss_name_base_index],
                                            float(sys.argv[3 + skip_loss_name_base_index]),
                                            float(sys.argv[4 + skip_loss_name_base_index]),
                                            int(sys.argv[5 + skip_loss_name_base_index]))
                values = [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps)]
                params = []
                for value in values:
                    _p = copy.deepcopy(p)
                    setattr(_p, vname, value)
                    params.append(_p)
        else:
            values = [i for i, _ in enumerate(params)]
            vname = "Index of parameter sets"

        loss_function = candidate_losses[loss_function_name]

        with torch.no_grad():
            if type(params[0]) is list:
                for _p in params[len(loss_values):]:
                    inner_loss_list = []
                    for i in range(steps2):
                        _, v_final, _ = RDGenerator.simulate_constant_steps(u.clone(), v.clone(), _p[i],
                                                                            num_steps=40000,
                                                                            disable_progress_bar=bool(i % 5))
                        inner_loss_list.append(loss_function(v_batch, v_final))
                    loss_values.append(inner_loss_list)
                    with open(path, "a") as f:
                        f.write(json.dumps(inner_loss_list) + "\n")

                data_array = np.array(loss_values) # 0-F, 1-k
                if three_d:
                    v1_vals = np.linspace(vmax, vmin, data_array.shape[0])
                    v2_vals = np.linspace(v2min, v2max, data_array.shape[1])
                    K, F = np.meshgrid(v2_vals, v1_vals)
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    surf = ax.plot_surface(K, F, data_array, cmap='viridis', edgecolor='none', alpha=0.9)
                    fig.colorbar(surf, ax=ax, shrink=0.5, label='Loss')
                    ax.set_xlabel(v2name)
                    ax.set_ylabel(vname)
                    ax.set_zlabel('Loss Value')
                    ax.set_title(f"Loss Landscape 3D ({v2name}-{vname})")
                    plt.tight_layout()
                else:
                    plt.imshow(data_array, cmap='viridis', extent=(v2min, v2max, vmin, vmax))  # or 'hot', 'plasma', 'gray'
                    plt.colorbar()
                    plt.xlabel(v2name)
                    plt.ylabel(vname)
                    plt.gca().set_aspect((v2max - v2min) / (vmax - vmin))
                    plt.title(f"Loss Landscape 2D Cross-Section ({v2name}-{vname})")
            else:
                compare_loss_funcs = False
                compare_pair = [candidate_losses['non-windowed'], candidate_losses['windowed']]
                for _p in params:
                    # _, v_final, _, _ = RDGenerator(params=_p).simulate_to_steady_trunc_bptt(up, vp, device=device,
                    #                                                                         tol=1e-8, max_steps=50000,
                    #                                                                         disable_progress_bar=False)
                    _, v_final, _ = RDGenerator.simulate_constant_steps(u.clone(), v.clone(), _p, num_steps=40000)
                    if compare_loss_funcs:
                        loss_values.append([l_func(v_batch, v_final) for l_func in compare_pair])
                    else:
                        loss_values.append(loss_function(v_batch, v_final))

                if compare_loss_funcs:
                    fig, ax = plt.subplots(figsize=(6, 6))

                    used_left = []
                    used_right = []
                    min_gap = 0.5
                    for lv in loss_values:
                        color = 'steelblue' if lv[0] > lv[1] else 'tomato'
                        ax.plot([0, 1], [lv[0], lv[1]], color=color, alpha=0.7, linewidth=2, marker='o', markersize=8)

                    for i, lv in sorted(enumerate(loss_values), key=lambda _x: _x[1][0]):
                        y_left = lv[0]
                        for y in used_left:
                            if abs(y_left - y) < min_gap:
                                y_left = y + min_gap
                        used_left.append(y_left)
                        ax.text(-0.1, y_left, f'{lv[0]:.1f}', ha='right', va='center', fontsize=9)
                        ax.text(-0.25, y_left, f'{i + 1:1d}', ha='right', va='center', fontsize=9)

                    for lv in sorted(loss_values, key=lambda _x: _x[1]):
                        y_right = lv[1]
                        for y in used_right:
                            if abs(y_right - y) < min_gap:
                                y_right = y + min_gap
                        used_right.append(y_right)
                        ax.text(1.08, y_right, f'{lv[1]:.1f}', ha='left', va='center', fontsize=9)

                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Non-windowed', 'Windowed'], fontsize=12)
                    ax.set_xlim(-0.3, 1.3)
                    ax.set_ylabel('Loss Value')
                    ax.set_title(
                        'Compare Two Loss Functions\n(blue: non-windowed > windowed\nred: windowed > non-windowed)')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    plt.tight_layout()
                else:
                    plt.plot(values, loss_values, marker='o', linestyle='-', color='b', lw=0.5, ms=3, markeredgewidth=0)
                    plt.xlabel(vname)
                    plt.ylabel('Loss')
                    plt.title('Loss Landscape')

            import os
            if os.path.exists('/kaggle/working'):
                plt.savefig('/kaggle/working/loss_landscape.png')
            elif os.path.exists('/content'):
                plt.savefig('/content/loss_landscape.png')
            else:
                plt.show()
        exit(0)

    # For 'generate targets', 'test training forward', and 'show stored targets' tasks only
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    v_batch_np = v_batch.detach().cpu().numpy().squeeze(1)

    for ax, img in zip(axes.flatten(), v_batch_np):
        ax.imshow(img, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("Range of pixel values:", v_batch.min().item(), v_batch.max().item(),
          "Median and average std of pixels:", v_batch.median().item(), v_batch.std(dim=(-1, -2)).mean().item())

