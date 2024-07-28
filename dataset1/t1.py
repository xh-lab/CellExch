import chardet

def convert_to_utf8(input_file_path, output_file_path):
   
    with open(input_file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']


    if encoding == 'ascii' or encoding is None:
        common_encodings = ['utf-8', 'gbk', 'windows-1252', 'iso-8859-1']
        for enc in common_encodings:
            try:
                with open(input_file_path, 'r', encoding=enc) as f:
                    content = f.read()
                encoding = enc
                break
            except UnicodeDecodeError:
                continue


    if encoding is None:
        with open(input_file_path, 'rb') as f:
            content = f.read()
        print("Could not detect encoding, proceeding with binary content.")


    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return encoding


input_path = "/home/jby2/XH/CellExch/dataset1/generation/original_LRI.csv"
output_path = "/home/jby2/XH/CellExch/dataset1/generation/origin_LRI.csv"
convert_to_utf8(input_path, output_path)