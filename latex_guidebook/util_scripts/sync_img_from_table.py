import json # handle ipynb
import base64 # convert image string to image
import os # joining file paths

BASE_IMG_DIR = "../sync_imgs/"
IMG_ext = ".png"


def get_cell_index_from_py(sync_id, c_path):
    code_strs = []
    code_flag = False
    target_idx = -1
    with open(c_path, 'r') as fh:
        data = json.load(fh)
        for idx, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    clean = line.rstrip()
                    if clean == "# {{{" + str(sync_id) + "}}}":
                        target_idx = idx
    return target_idx


def get_img_str(c_path, target_index):
    with open(c_path, 'r') as fh:
        data = json.load(fh)
        img_str = data['cells'][target_index]['outputs'][0]['data']['image/png']
    return img_str


def convert_and_write(img_str, img_path):
    img_data = base64.b64decode(img_str)
    with open(img_path, 'wb') as fh:
        fh.write(img_data)
        return True
    return False


def create_image_path(sync_id):
    # make dir/sub_dir as necessary
    global BASE_IMG_DIR

    id_split = sync_id.split("_")
    img_dir = id_split[0]
    img_path = os.path.join(BASE_IMG_DIR, img_dir)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if len(id_split) == 2:
        img_path = os.path.join(img_path, id_split[1] + IMG_ext)
    elif len(id_split) == 3:
        img_subdir = id_split[1]
        img_path = os.path.join(img_path, img_subdir)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_path = os.path.join(img_path, id_split[2] + IMG_ext)
    else:
        print("sync id > 1 subdir, defaulting to {}".format(id_split[1]))
        img_subdir = id_split[1]
        img_path = os.path.join(img_path, img_subdir)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_path = os.path.join(img_path, id_split[-1] + IMG_ext)
    
    return img_path

def create_latex_fig_entry(img_path, sync_id, width=0.6, caption=None):
    """Creates a figure block for latex
    
    Args:
        img_path: path to image file
            
    Returns:
        Latex figure code lines. The code block will be empty if no code was 
        passed in.
    """

    code_block = []
    code_block.append("\\begin{figure}\n")
    code_block.append("\\centering\n")
    code_block.append("\\includegraphics[width=" + width + "\\textwidth]{" + img_path + "}\n")
    if caption:
        code_block.append("\\caption{" + caption + "}\n")
    code_block.append("\\label{fig:" + sync_id + "}\n")
    code_block.append("\\end{figure}\n")

    return code_block

def create_new_tex_file_data(sync_id, l_path, code_block):
    """create the formatted data including the latex block to write to the .tex file
    
    Args:
        sync_id: the id of the code block being synched
        l_path: String path to the tex file
        code_block: the block of code to write to the file
        language: the language to specify for latex

    Returns:
        data: "correctly" created latex data
    """

    # TODO: consider restructuring this function

    with open(l_path, 'r') as fh:
        data = fh.readlines()
        idx = 0
        prev_block_stop = -1
        for idx, line in enumerate(data):
            clean = line.strip()
            if clean == "% {{{" + str(sync_id) + "}}}":
                # will insert code block on next line
                code_idx = idx+1
                break

        # handling a pre-existing code block vs new code block
        if data[code_idx].strip() == "\\begin{figure}":
            end_found = False
            for idx, line in enumerate(data[code_idx:]):
                clean = line.strip()
                if clean.strip() == "\\end{figure}":
                    end_found = True
                    prev_block_stop = idx+1
                    break
            
            if end_found:
                data[code_idx: code_idx+prev_block_stop] = code_block
            else:
                print("ERROR!, no stop code found for code block {}".format(sync_id))

        else:
            # embed new code block
            data[code_idx: code_idx] = code_block
    
    if prev_block_stop >= 0:
        return data, code_idx+prev_block_stop
    else:
        return data, code_idx+len(code_block)

def sync_snippet(sync_id, c_path, l_path, opts):

    img_path = create_image_path(sync_id)
    target_index = get_cell_index_from_py(sync_id, c_path)
    if target_index >= 0:
        img_str = get_img_str(c_path, target_index)
        success = convert_and_write(img_str, img_path)
        if not success:
            print("Error syncing {}".format(sync_id))
    else:
        print("Error: no img data for indicated cell")
    
    # handle options
    #print(code_block)
    img_w = 0.6
    opts = opts.split()
    for o in opts:
        if o.startswith("width"):
            img_w = o.split("-")[1]

    # create latex information
    # need to remove `.` from img path since we need to be relative to main.tex
    # not this file anymore
    img_path = img_path[1:]
    code_block = create_latex_fig_entry(img_path, sync_id, width=img_w, caption=None)
    new_data, _ = create_new_tex_file_data(sync_id, l_path, code_block)
    
    # ===== write to file
    if new_data:
        # Write to file: overwrite existing file with modifications
        with open(l_path, 'w') as fh:
            fh.writelines(new_data)
        print("Completed: {}".format(sync_id))


def read_sync_table(table_file):
    with open(table_file) as fh:
        for line in fh:
            cols = [col.strip() for col in line.split("||")]
            sync_snippet(sync_id=cols[0], c_path=cols[1], 
                         l_path=cols[2], opts=cols[3])

def main():
    read_sync_table(table_file = "./img_sync_table.txt")

if __name__ == "__main__":
    main()