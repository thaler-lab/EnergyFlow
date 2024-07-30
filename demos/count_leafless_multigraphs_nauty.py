import argparse
from collections import Counter
import math
import os
import subprocess
import time

from scipy.special import binom

import energyflow as ef

# print spaces every three digits of a number
def format_number(n):
    s = str(n)[::-1]
    return ' '.join([s[3*i:3*i+3] for i in range(math.ceil(len(s)/3))])[::-1]

# function to count the number of leafless, non-isomorphic multigraphs with up to maxd edges
def count_leafless_multigraphs(maxd, nauty_path):

    # iterate over dmax
    for dmax in range(1, maxd+1):
        print('dmax =', dmax)

        # generate non-isomorphic simple graphs with up to dmax edges
        start = start0 = time.time()
        b = bytes()
        for n in range(2, dmax+2):
            command = '{} -cd1 {} 0:{}'.format(os.path.join(nauty_path, 'geng'), n, dmax)
            result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            b += result.stdout
        print('Running geng took {:.3f}s to generate simple graphs'.format(time.time() - start))

        # running multig
        start = time.time()
        result = subprocess.run('{} -T -e0:{}'.format(os.path.join(nauty_path, 'multig'), dmax).split(),
                                input=b, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('Running multig took {:.3f}s to generate multigraphs'.format(time.time() - start))

        # processing multig output
        start = time.time()
        ncon = (dmax+1)*[0]
        for line in str(result.stdout, 'ascii').split('\n'):
            if not line:
                continue
            parts = line.split()
            intparts = list(map(int, parts))
            n = intparts[0]
            d = sum(intparts[4::3])
            vals = n*[0]
            gps = intparts[2:]
            for i in range(len(gps)//3):
                v1,v2,m = gps[3*i:3*(i+1)]
                vals[v1] += m
                vals[v2] += m
            if min(vals) > 1:
                ncon[d] += 1

        # combinatorializing
        integer_parts = {n: list(ef.algorithms.int_partition_unordered(n)) for n in range(dmax+1)}
        nall = [0]
        for d in range(1, dmax+1):
            tot = 0
            for intpart in integer_parts[d]:
                term = 1
                for dp,r in Counter(intpart).items():
                    term *= binom(ncon[dp] + r - 1, r)
                tot += term
            nall.append(int(tot))
        print('Counting multigraphs took {:.3f}s'.format(time.time() - start))

        # print results
        lenmaxd, lenmaxncon, lenmaxnall = len(format_number(dmax)), max(len(format_number(ncon[-1])), 9), max(len(format_number(nall[-1])), 3)
        line = '{}{}{}{}{}{}{}'.format('| {:>', lenmaxd, '} | {:>', lenmaxncon, '} | {:>', lenmaxnall, '} |')
        header = line.format('d', 'Connected', 'All')
        print('Number of leafless, non-isomorphic multigraphs')
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for d in range(1, dmax+1):
            print(line.format(format_number(d), format_number(ncon[d]), format_number(nall[d])))
        print('-'*len(header))
        print('Total time: {:.3f}s'.format(time.time() - start0))
        print()

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Counts non-isomorphic, leafless multigraphs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--maxd', '-d', type=int, default=10, help='maximum number of edges')
    parser.add_argument('--nauty-path', '-np', default='/home/pkomiske/opt/nauty27rc4', help='path to nauty install')
    args = parser.parse_args()

    # do the counting
    count_leafless_multigraphs(args.maxd, args.nauty_path)
