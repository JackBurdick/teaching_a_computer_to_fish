import json
# sync python files


def create_latex_entry(code_list, language="pyInStyle"):
    """Creates a python-like code block for latex
    
    Args:
        code_list: List of code lines
            
    Returns:
        List of code lines formated/wrapped for latex. The code block will be 
        empty if no code was passed in.
    """
    code_block = []
    code_block.append("\\begin{lstlisting}[style="+language+"]\n")
    code_block.extend(code_list)
    code_block.append("\\end{lstlisting}\n")
    
    return code_block

def create_new_tex_file_data(sync_id, l_path, code_block, language="pyInStyle"):
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
        if data[code_idx].strip() == "\\begin{lstlisting}[style="+language+"]":
            end_found = False
            for idx, line in enumerate(data[code_idx:]):
                clean = line.strip()
                if clean.strip() == "\\end{lstlisting}":
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
                        target_idx = idx
                    elif clean == "# END}}}":
                        code_flag = False
                
                    if code_flag:
                        # add newline char for writing to tex file
                        code_strs.append(clean + "\n")
    
    # remove first code string ("# {{{" + str(sync_id)
    return code_strs[1:], target_idx


def get_cell_output(c_path, target_index):
    with open(c_path, 'r') as fh:
        data = json.load(fh)
        # this is a "magic path" that is specific to .ipynb
        target_output = data['cells'][target_index]['outputs'][0]['text']
    return target_output


def include_output_block(sync_id, data, fmt_out_block, entry_point, language="pyOutStyle"):
    """create the formatted data including the latex block to write to the .tex file
    
    Args:
        sync_id: the id of the code block being synched
        data: tex file data
        fmt_out_block: the block of code to write to the file
        language: the language to specify for latex

    Returns:
        data: "correctly" created latex data
    """

    # go to next line
    entry_point += 1

    # handling a pre-existing code block vs new code block
    if data[entry_point].strip() == "\\begin{lstlisting}[style="+language+"]":
        end_found = False
        for idx, line in enumerate(data[entry_point+1:]):
            clean = line.strip()
            if clean.strip() == "\\end{lstlisting}":
                end_found = True
                block_stop_loc = idx+1
                break
        
        if end_found:
            data[entry_point: entry_point+block_stop_loc+1] = fmt_out_block
        else:
            print("ERROR!, no stop code found for code block {}".format(sync_id))

    else:
        # embed new code block
        data[entry_point: entry_point] = fmt_out_block
    return data


def sync_snippet(sync_id, c_path, l_path, opts):
    code_strs, target_index = get_code_from_py(sync_id, c_path)

    fmt_code_block = create_latex_entry(code_list=code_strs, language="pyInStyle")
    new_data, prev_block_stop = create_new_tex_file_data(sync_id, l_path, 
                                                         fmt_code_block, 
                                                         language="pyInStyle")

    # handle options
    opts = opts.split()
    if "o" in opts:
        if target_index >= 0:
            output_strs = get_cell_output(c_path, target_index)
            fmt_out_block = create_latex_entry(code_list=output_strs, 
                                               language="pyOutStyle")
            new_data = include_output_block(sync_id, new_data, fmt_out_block, 
                                            prev_block_stop, language="pyOutStyle")
        else:
            print("Error: no ouptput for indicated cell")

    #print(new_data)
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
    read_sync_table(table_file = "./sync_table.txt")

if __name__ == "__main__":
    main()