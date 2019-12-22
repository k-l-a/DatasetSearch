#!/usr/bin/env python

"""
Convert CSV file to libsvm format. Works only with numeric variables.
Put -1 as label index (argv[3]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[4] == 1.
"""

import sys
import csv
from collections import defaultdict

def construct_line( label, line ):
	new_line = []
	try:
		if float( label ) == 0.0:
			label = "0"
	except ValueError:
		label = "0"

	new_line.append( label )

	for i, item in enumerate( line ):
		try:
			if item == '' or float(item) == 0.0:
				continue
		except ValueError:
			#Skip categorical
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

def convert(input, output, label_index, skip_header):
	input_file = input
	output_file = output

	try:
		label_index = int( label_index )
	except IndexError:
		label_index = 0

	try:
		skip_headers = skip_header
	except IndexError:
		skip_headers = 0

	i = open( input_file, 'rt' )
	o = open( output_file, 'wt' )

	reader = csv.reader(i)

	if skip_headers:
		headers = next(reader, None)

	for line in reader:
		if label_index == -1:
			label = '1'
		else:
			label = line.pop( label_index )

		try:
			new_line = construct_line( label, line )
			o.write( new_line )
		except ValueError:
			#do nothing
			new_line = ""