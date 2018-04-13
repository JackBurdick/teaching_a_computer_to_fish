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
 
def sync_snippet(sync_id, c_path, l_path, opts):

    img_path = create_image_path(sync_id)
    target_index = get_cell_index_from_py(sync_id, c_path)
    if target_index >= 0:
        img_str = get_img_str(c_path, target_index)
        success = convert_and_write(img_str, img_path)
        if success:
            print("Completed {}".format(sync_id))
        else:
            print("Error syncing {}".format(sync_id))
    else:
        print("Error: no img data for indicated cell")
    #opts = opts.split()
    #if "o" in opts:


def read_sync_table(table_file):
    with open(table_file) as fh:
        for line in fh:
            cols = [col.strip() for col in line.split("||")]
            sync_snippet(sync_id=cols[0], c_path=cols[1], 
                         l_path=cols[2], opts=cols[3])


def main():
    read_sync_table(table_file = "./img_table.txt")

if __name__ == "__main__":
    main()