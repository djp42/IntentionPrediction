#this is a python script to find the orders the features were selected in

def find_order(log_file):
    for line in log_file.readlines():
        if "Ordered" in line:
            print(line)
