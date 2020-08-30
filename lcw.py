# -*- coding: utf-8 -*-

from argparse import *
import sys, struct, base64, io
import numpy

class MyFormatter(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass

def LCWpack(Source):
    packed_buffer = []
    i = 0
    while i < len(Source):
        l = min(len(Source) - i, 63)
        packed_buffer.append(0x80 + l)
        packed_buffer += list(Source[i:i+l])
        i += l
    packed_buffer.append(0x80)
    return struct.pack("HH", len(packed_buffer), len(Source)) + bytes(packed_buffer)

def base64write(data, f):
    encoded = base64.b64encode(data).decode(encoding='ascii')
    i = 0
    for j in range(0, len(encoded), 70):
        i += 1
        print(i, encoded[j:j+70], file=f, sep='=')

def LCWunpack(Source, Dest):
    SP = 0
    DP = 0
    rel = Source[SP]
    if rel == 0:
        SP += 1
    while True:
        Com = Source[SP]
        SP += 1
        b7 = Com >> 7
        if b7 == 0:
            # copy command (2)
            # Count is bits 4-6 + 3
            Count = ((Com & 0x7F) >> 4) + 3
            # Position is bits 0-3, with bits 0-7 of next byte
            Posit = ((Com & 0x0F) << 8) + Source[SP];
            SP += 1
            # Starting pos=Cur pos. - calculated value
            Posit = DP - Posit
            for i in range(Posit, Posit+Count):
                Dest[DP] = Dest[i]
                DP += 1
        elif b7 == 1:
            # Check bit 6 of Com
            b6 = (Com & 0x40) >> 6
            if b6 == 0:
                # Copy as is command (1)
                Count = Com & 0x3F # mask 2 topmost bits
                if Count == 0:
                    break # EOF marker
                for _ in range(Count):
                    Dest[DP] = Source[SP]
                    DP += 1
                    SP += 1
            if b6 == 1: # large copy, very large copy and fill commands
                # Count = (bits 0-5 of Com) +3
                # if Com=FEh then fill, if Com=FFh then very large copy
                Count = Com & 0x3F
                if Count < 0x3E: # large copy (3)
                    Count += 3
                    # Next word = pos. from start of image
                    pos = struct.unpack("H", Source[SP:SP+2])[0]
                    if rel == 0: # relative
                        Posit = DP - pos
                    else:
                        Posit = pos
                    SP += 2
                    for i in range(Posit, Posit+Count):
                        Dest[DP] = Dest[i]
                        DP += 1
                elif Count == 0x3F: # very large copy (5)
                    # next 2 words are Count and Pos
                    Count = struct.unpack("H", Source[SP:SP+2])[0]
                    pos = struct.unpack("H", Source[SP+2:SP+4])[0]
                    if rel == 0: # relative
                        Posit = DP-pos
                    else:
                        Posit = pos
                    SP += 4
                    for i in range(Posit, Posit+Count):
                        Dest[DP] = Dest[i]
                        DP+= 1
                else:
                    # Count == 0x3E, fill (4)
                    # Next word is count, the byte after is color
                    Count = struct.unpack("H", Source[SP:SP+2])[0]
                    SP+= 2
                    b = Source[SP]
                    SP += 1
                    for i in range(Count):
                        Dest[DP] = b
                        DP += 1

def main(args):
    with open(args.filename, "r") as f:
        base64text = ""
        line = ""
        while line != args.section:
            line = f.readline().strip()
        line = f.readline().strip()
        i = 1
        try:
            while int(line.split("=", 1)[0]) == i:
                base64text += line.split("=", 1)[1]
                line = f.readline().strip()
                i += 1
        except:
            pass
    content = base64.b64decode(base64text, validate=True)
    if args.lcw:
        with io.BytesIO(content) as f:
            while True:
                header = f.read(4)
                if len(header) != 4:
                    break
                packed_size, unpacked_size = struct.unpack("HH", header)
                print(packed_size, unpacked_size, file=sys.stderr)
                packed_buffer = f.read(packed_size)
                assert(len(packed_buffer) == packed_size)
                unpacked_buffer = [0] * unpacked_size
                LCWunpack(packed_buffer, unpacked_buffer)
                sys.stdout.buffer.write(bytes(unpacked_buffer))
    else:
        sys.stdout.buffer.write(content)
    return 0

def extract_tiles(data, ini):
    total_len = len(data)
    if ini == 0 or ini == 1 or ini == 2:
        data = numpy.frombuffer(data, dtype=numpy.uint8)
        templates = data.reshape((-1, 3))[:, [0,1]]
        icons = data[2::3]
    else:
        templates = numpy.frombuffer(data[:2*total_len//3], dtype=numpy.uint16)
        icons = numpy.frombuffer(data[2*total_len//3:], dtype=numpy.uint8)
    templates = templates.reshape((128, 128))
    icons = icons.reshape((128, 128))
    
    return templates
    # print(icons)    

if __name__ == "__main__":
    parser = ArgumentParser(description='Extract LCW Packed content from text file.\n'
        'Author: Gábor Borbély (gaebor).\n\n'
        'With the help of the followings:\n'
        'https://cnc.fandom.com/wiki/Red_Alert_File_Formats_Guide\n'
        'http://www.shikadi.net/moddingwiki/Westwood_LCW\n\n'
        '1. Locate a section\n2. concatenate base64 chunks\n3. base64 decode\n'
        '4. LCW decode',
        formatter_class=MyFormatter)
        
    parser.add_argument("filename", type=str)
    parser.add_argument("--section", "-s", dest="section", default="[MapPack]",
        help="which section to extract", type=str)
    parser.add_argument("-L", dest="lcw", default=True,
        help="don't LCW decompress", action="store_false")
    parser.add_argument("--ini", "-i", dest="ini", default=3,
        help="NewINIFormat", type=int)
    # parser.add_argument("-n", "--max", dest="n", default=128,
        # help="max dimension of map", type=int)
    # parser.add_argument("-x", "-X", "--x", "--X", dest="x", default=1,
        # help="map X", type=int)
    # parser.add_argument("-y", "-Y", "--y", "--Y", dest="y", default=1,
        # help="map Y", type=int)
    # parser.add_argument("-w", "--width", "-W", "--Width", dest="width", default=126,
        # help="map Width", type=int)
    # parser.add_argument("-H", "--height", "--Height", dest="height", default=126,
        # help="map Height", type=int)
        
    sys.exit(main(parser.parse_args()))
