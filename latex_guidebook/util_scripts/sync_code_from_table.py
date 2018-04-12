import json
# sync python files


def create_latex_entry(code_strs):
    """Creates a python-like code block for latex
    
    Args:
        code_strs: List of code lines
            
    Returns:
        List of python code lines formated/wrapped for latex. The code block 
        will be empty if no code was passed in.
    """
    code_block = []
    code_block.append("\\begin{lstlisting}[language=Python]\n")
    code_block.extend(code_strs)
    code_block.append("\\end{lstlisting}\n")
    return code_block

def create_new_tex_file_data(sync_id, l_path, opts, code_block):
    """write the formatted latex block to the .tex file
    
    Args:
        sync_id: the id of the code block being synched
        l_path: String path to the tex file
        opts: any formatting options
        code_block: the block of code to write to the file

    Returns:
        data: "correctly" created latex data
    """

    # TODO: consider restructuring this function

    with open(l_path, 'r') as fh:
        data = fh.readlines()
        idx = 0
        for idx, line in enumerate(data):
            clean = line.strip()
            if clean == "% {{{" + str(sync_id) + "}}}":
                code_idx = idx
                break

        # handling a pre-existing code block vs new code block
        if data[code_idx+1].strip() == "\\begin{lstlisting}[language=Python]":
            end_found = False
            for idx, line in enumerate(data[code_idx:]):
                clean = line.strip()
                if clean.strip() == "\\end{lstlisting}":
                    end_found = True
                    prev_block_stop = idx
                    break
            
            if end_found:
                data[code_idx+1: code_idx+1+prev_block_stop] = code_block
            else:
                print("ERROR!, no stop code found for code block {}".format(sync_id))

        else:
            # embed new code block
            data[code_idx+1: code_idx+1] = code_block
    return data


def get_code_from_py(sync_id, c_path):
    code_strs = []
    code_flag = False
    with open(c_path, 'r') as fh:
        data = json.load(fh)
        for idx, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    clean = line.rstrip()
                    if clean == "# {{{" + str(sync_id):
                        code_flag = True
                    elif clean == "# END}}}":
                        code_flag = False
                
                    if code_flag:
                        # add newline char for writing to tex file
                        code_strs.append(clean + "\n")
    
    # remove first code string ("# {{{" + str(sync_id)
    return code_strs[1:], idx
  

def sync_snippet(sync_id, c_path, l_path, opts):
    code_strs, target_index = get_code_from_py(sync_id, c_path)
    fmt_code_block = create_latex_entry(code_strs)
    
    new_data = create_new_tex_file_data(sync_id, l_path, opts, fmt_code_block)
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
    read_sync_table(table_file = "./sync_table.txt")

if __name__ == "__main__":
    main()