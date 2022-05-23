import csv
import collections
import textwrap 
import hashlib

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import numpy as np

import distutils.util
import argparse

def strtobool(v):
    return bool(distutils.util.strtobool(v))

def is_float(v):
    try:
        float(v)
        return True
    except:
        return False

def get_parser():
    parser = argparse.ArgumentParser(
        description='Result Plotting Script')
    parser.add_argument('--y_key', default='f1_macro', type=str,
        help='the field name (key) of the value to be plotted along the y-axis')
    parser.add_argument('--x_key', default=None, type=str,
        help='the field name (key) of the value to be plotted along the y-axis\n\
            when it is None, a dynamic selection is performed')
    # parser.add_argument('--aggregate_keys', default=['held_out'], type=str,
    #     help='the field names (keys) of the value to be used for aggregation across runs\n\
    #         when it is None, a dynamic selection is performed')
    parser.add_argument('--exclude_keys',  default=['f1_weighted', 'held_out'], type=str, nargs='*',
        help='the field names (keys) of values to be exclude when grouping entries')
    parser.add_argument('--top_n_lines',  default=2, type=int,
        help='the number of lines with the highest values to be plotted per category (see below), if less than or equal to 0, all lines are plotted')
    parser.add_argument('--top_n_separate_pipeline',  default=True, type=strtobool,
        help='enable separate counters for number of lines plots for every pipeline (treat pipelines as separate categories)')
    parser.add_argument('--top_n_x_coor',  default=None, type=str,
        help='the x-coordinate for the deciding top-n lines, \n\
            if is None, every x-coordinate will be treated separately (i.e. top-n at every x-coordinate will be plotted)')
    parser.add_argument('--top_n_aggregate',  default=None, choices=['avg_y', 'max_y', 'none'], type=str,
        help='the aggregate measure across x-axis for the deciding top-n lines, \n\
            this overrides the top_n_x_coor argument if is not "none" or None')
    parser.add_argument('--show_label',  default=True, type=strtobool,
        help='displays the value as text on the plot if True')
    parser.add_argument('--show_error_bar',  default=True, type=strtobool,
        help='displays the error bar for aggregated results')
    parser.add_argument('--consistent_color',  default=False, type=strtobool,
        help='uses color that is consistent for each particular setting across different plots')
    parser.add_argument('--consistent_color_salt',  default='random_color', type=str,
        help='a salt which changes the colors generated')
    parser.add_argument('--figsize',  default='(16, 12)', type=str,
        help='the figsize argument to a matplotlib figure')
    parser.add_argument('--out_file',  default='plots.png', type=str,
        help='the path to which the plot is saved (Note: different extensions such as .pdf and .png are supported by the matplotlib library)')
    
    return parser

if __name__ == '__main__':
    
    sns.set_context('poster')

    mpl.rc('pdf', fonttype=42)
    mpl.rc('ps', fonttype=42)

    parser = get_parser()
    
    # Parses the csv files using unknownargs 
    args, unknownargs = parser.parse_known_args()

    # This separates all unknown args into pairs. The first (should come with dashes at the beginning) is treated as the pipeline, and the second is assumed to be the path to the csv file
    file_paths_iter = iter(unknownargs)
    file_paths = list([(x.lstrip('-'), next(file_paths_iter)) for x in file_paths_iter])
    print(file_paths)
    
    grouped_lines = collections.defaultdict(list) # This is the collection of lines to be plotted
    y_key = args.y_key
    exclude_keys = args.exclude_keys

    legend_key_str = ''
    for tag, file_path in file_paths:

        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            all_rows = list(reader)
        
        
        x_key = args.x_key 
        if x_key is None:
            x_key = 'fine_tune_take' if 'supervised' not in tag else 'take'
        

        for row in all_rows:
            
            # Each row contains a dictionary of key-value pairs. To ensure the order of traversal across runs, sorting is performed.
            sorted_row = sorted(row.items(), key=lambda item: item[0])
                
            # Generates the final entry in the legend which shows the correspondence of hyperparameters
            key_string_entries = [k for k, v in sorted_row]
            key_string = '| '.join(['pipeline'] + [f"{i}:{entry}" for i, entry in enumerate(key_string_entries)])
            if len(key_string) > len(legend_key_str):
                legend_key_str = key_string
            
            # This generates a unique string for a particular setting, in order to group entries into the same line.
            # exclude_keys are effectively used to create groupings. By ignoring different keys, numbers from different runs can be grouped 
            # according to the identifier string. 
            # The number for the x-axis, y-axis, and exclude_keys are replaced with placeholders such that their values will not be used in the identifier string
            run_key_entries = [
                'x' if k == x_key else
                'y' if k == y_key else
                'e' if k in exclude_keys else
                # 'a' if k in aggregate_keys else
                str(v)
                    for k, v in sorted_row
                ]

            run_key = '| '.join([tag] + [f"{i}:{entry}" for i, entry in enumerate(run_key_entries)])
            grouped_lines[run_key].append((row[x_key], row[y_key]))
    # print(grouped_lines)

    top_n_x_coor = args.top_n_x_coor

    # This post-process the numbers to perform aggregation
    for k in grouped_lines:
        line_xs = np.unique([x for x, y in grouped_lines[k]])
        aggregated_list = []
        
        # For each x coordinate, all values of y are aggregated
        for x_aggregate in line_xs:
            ys = [float(y) for x, y in grouped_lines[k] if x == x_aggregate]
            aggregated_list.append((x_aggregate, np.mean(ys), np.std(ys)))
        grouped_lines[k] = aggregated_list

        # This sorts the x coordinate for final plotting
        if all([is_float(item[0]) for item in grouped_lines[k]]):
            grouped_lines[k] = sorted([(float(item[0]), item[1], item[2]) for item in grouped_lines[k]], key=lambda item: item[0])
            if top_n_x_coor is not None:
                top_n_x_coor = float(top_n_x_coor)
        else:
            grouped_lines[k] = sorted(grouped_lines[k], key=lambda item: item[0])
    
    top_n_lines = args.top_n_lines
    top_n_separate_pipeline = args.top_n_separate_pipeline
    top_n_aggregate = args.top_n_aggregate
    show_label = args.show_label
    consistent_color = args.consistent_color
    consistent_color_salt = args.consistent_color_salt

    lines_drawing_data = {}
    # This dictionary contains the drawn boolean which controls whether a line is plotted
    for k in grouped_lines:
        lines_drawing_data[k] = {
            'drawn': False if top_n_lines > 0 else True,
            'tag': k.split('|')[0]
        }

    unique_tags = np.unique([k for k, v in file_paths])

    # Do not perform filtering of lines when top_n_lines is > 0
    if top_n_lines > 0:
        
        # For each setting, select the associated lines and perform filtering
        for tag in unique_tags:

            # Decide whether to separate different pipelines or not
            if top_n_separate_pipeline:
                lines_with_tag = dict(filter(lambda item: item[0].startswith(tag + '|'), grouped_lines.items()))
            else:
                lines_with_tag = grouped_lines
            
            # all_xs contains all the x coordinates to check for rankings
            # top_n_x_coor is not None, the top_n_x_coor will be the sole coordinate
            all_xs = list(np.unique([x for v in lines_with_tag.values() for x, y, std in v]))
            # print(top_n_x_coor, all_xs, top_n_x_coor in all_xs)
            if top_n_x_coor is not None and top_n_x_coor in all_xs:
                all_xs = [top_n_x_coor]

            if top_n_aggregate is not None and top_n_aggregate != 'none':
                if top_n_aggregate == 'max_y':
                    # Find the maximum of the whole line
                    ys = [(np.amax([y for x, y, std in v]), k) for k, v in lines_with_tag.items()]
                else:
                    # Find the average of the whole line
                    ys = [(np.mean([y for x, y, std in v]), k) for k, v in lines_with_tag.items()]

                sorted_ys = sorted(ys, key=lambda item: float(item[0]), reverse=True)
                for y, k in sorted_ys[:top_n_lines]:
                    # Set the drawing flag for top n lines to True
                    lines_drawing_data[k]['drawn'] = True
                    lines_drawing_data[k][top_n_aggregate] = y
            else:

                for x_coor in all_xs:
                    # Find all y values which has the x_coor
                    ys = [(y, k) for k, v in lines_with_tag.items() for x, y, std in v if x == x_coor]
                    sorted_ys = sorted(ys, key=lambda item: float(item[0]), reverse=True)
                    for y, k in sorted_ys[:top_n_lines]:
                        # Set the drawing flag for top n lines to True
                        lines_drawing_data[k]['drawn'] = True

            if not top_n_separate_pipeline:
                # break if different pipelines are not differentiated
                break

    

    all_styles = [
        'o','^','s','P','*','X','d','v','<','>','1','2','3','4','8','p'
    ]
    tag_style = {}
    for i, tag in enumerate(unique_tags):
        # Marker styles for different pipelines
        tag_style[tag] = all_styles[i % len(all_styles)]

    figsize = eval(args.figsize)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0])

    # Generate consistent color for runs by hashing
    for k in sorted(grouped_lines.keys()):
        if consistent_color:
            hash_string = k + consistent_color_salt * 2
            color_hash = int(hashlib.sha256(hash_string.encode('utf-8')).hexdigest(), 16) % 256 ** 3
            hsv = (color_hash % 64 / 64,  (color_hash // 64 ** 2 % 64) / 64, 0.75)
            color = mpl.colors.hsv_to_rgb(hsv)
        else:
            color = None

        xs = np.array([x for x, y, std in grouped_lines[k]], dtype=float)
        ys = np.array([y for x, y, std in grouped_lines[k]], dtype=float)
        stds = np.array([std for x, y, std in grouped_lines[k]], dtype=float)
        if lines_drawing_data[k]['drawn']:
            if top_n_aggregate is not None and top_n_aggregate in lines_drawing_data[k]:
                label = k + f'| {top_n_aggregate}:{lines_drawing_data[k][top_n_aggregate]:.4f}'
            else:
                label = k
            if args.show_error_bar:
                ax.errorbar(xs, ys, stds,
                    label='\n'.join(textwrap.wrap(label, 50)), 
                    marker=tag_style[lines_drawing_data[k]['tag']], 
                    markersize=12, 
                    linestyle='-',
                    color=color,
                    capsize=5, 
                    capthick=2
                )
            else:
                ax.plot(xs, ys, 
                    label='\n'.join(textwrap.wrap(label, 50)), 
                    marker=tag_style[lines_drawing_data[k]['tag']], 
                    markersize=12, 
                    linestyle='-',
                    color=color
                )
            
            # Display labels on the plots
            if show_label:
                for x, y in zip(xs, ys):
                    text = ax.text(x, y, f'{y:.4f}', fontsize=16)
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=(1, 1, 1, 0.5)),
                        path_effects.Normal()])
    
    # Adds the entry of the legend
    patch = matplotlib.patches.Patch(color=(0.9,0.9,0.9), label='\n'.join(textwrap.wrap(legend_key_str, 50)))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[patch] + handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    ax.set_ylabel(y_key)
    ax.set_xlabel(x_key if args.x_key is not None else 'fine_tune_take')
    ax.grid()

    fig.savefig(args.out_file, bbox_inches = "tight")
    fig.show()


