from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

def replace(file_path, args):
    #Create temp file
    fh, abs_path = mkstemp()
    c = 0
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if line.startswith('default'):
                    l = line.split('=')
                    newline = l[0] + '= ' + str(args[c]) + '\n'
                    c = c + 1
                    new_file.write(newline)
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def main():
    replace('./autosklearn/pipeline/components/classification/gradient_boosting.py', [0.008476011455059052, 0.09977829739451409, 356.55484375, 202.45919181823731, 28.330279693603515])



main()
