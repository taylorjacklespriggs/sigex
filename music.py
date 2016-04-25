#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  main.py
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  

from signal_extrapolator import SignalExtrapolator,EXAMPLE_PARAMETERS
import sys
import numpy as np
import struct

def main(args):
	try:
		signals = []
		filenames = args[1:]
		dt = np.dtype(np.int16)
		dt = dt.newbyteorder('<')
		for fl in (open(fn, 'rb') for fn in filenames):
			signals.append(np.frombuffer(fl.read(), dtype=dt)\
				.astype(np.float32)/2**15)
			fl.close()
		sigex = SignalExtrapolator(EXAMPLE_PARAMETERS['music'])
		sigex.train(signals)
		lb = sigex.get_look_back()
		recons = np.random.rand(lb).astype(np.float32)*2-1
		stdout = open('/dev/stdout', 'wb')
		def disp(vals):
			for v in vals:
				stdout.write(struct.pack('<h', int(2**15*v)))
		# disp(recons)
		while True:
			ext = sigex.extrapolate(recons[-lb:])
			recons = np.concatenate((recons, ext))
			disp(ext)
	except Exception as e:
		print(str(e), file=sys.stderr)
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
