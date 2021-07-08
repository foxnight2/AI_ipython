# https://github.com/BVLC/caffe/blob/master/python/draw_net.py


import caffe_pb2
from google.protobuf import text_format

net = caffe_pb2.NetParameter()
text_format.Merge(open('./config.prototxt', 'rb').read(), net)




def load_protobuf_from_file(container, filename):
    with open(filename, 'rb') as fin:
        file_content = fin.read()

    # First try to read it as a binary file.
    try:
        container.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return container
    except Exception as e:  # pylint: disable=broad-except
        print ("Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format
        text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return container

