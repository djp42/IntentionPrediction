import collections, itertools, copy
import numpy, scipy, math, random
import os, sys, time, importlib
import tokenize, re, string
import json, unicodedata

# Import Constants class defined in "./lib/constants.py"
# need to create a Constants object with an initialization
# argument that is the path of the folder __main__.py is in
# because of some weird command line thing
import lib.constants as c
import bin
c.init(os.path.dirname(os.path.realpath(__file__)))

##
# Note to Bruno/Alex
# ------------------
# Change this variable to change which executable .py file you want to run
# if you do not want to specify that file name from the command line.
# Choices include "process_recipes", "query_online_db", "write_recipes"
##
DEFAULT_EXE_CHOICE = "setup"

##
# Function: main
# --------------
# argv = ["main.py", <executable>, <args..>]
# <executable> is one of: "process_recipes", "query_online_db", "write_recipes"
def main(argv):
	print("In __main__.py, c.PATH_TO_ROOT: ", c.PATH_TO_ROOT)
	print("In __main__.py, os.path.dirname(__file__): ", os.path.dirname(os.path.realpath(__file__)))

	exe_name = argv[c.EXE_ARG_POS]
	print("exe_name: ", exe_name)
	exe = importlib.import_module("." + exe_name, "bin")
	new_argv = [exe_name + ".py"] + argv[c.EXE_ARG_POS+1:]

	exe.main(new_argv)


if __name__ == "__main__":

	sys.path.append(c.PATH_TO_ROOT)

	argv = sys.argv

	# If no executable was provided, use the default
	if len(argv) < c.EXE_ARG_POS+1:
		raise Exception("No valid executable")

	# If an executable was provided that doesn't exist, use the default
	elif argv[c.EXE_ARG_POS] not in c.EXECUTABLES:
		raise Exception("No valid executable")
	#argv[c.EXE_ARG_POS] = DEFAULT_EXE_CHOICE

	main(argv)
