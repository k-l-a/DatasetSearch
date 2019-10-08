# importing os module
import os


# Function to rename multiple files
def main():

    for filename in os.listdir("OpenMlDatasets"):
        words = filename.split("_")
        if words[-2] == "None" or words[-2].startswith('['):
            new_name = "".join(words[:-2]) + words[-1]
            os.rename(filename, new_name)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()