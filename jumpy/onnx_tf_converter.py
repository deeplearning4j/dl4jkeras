from tensorflow.python.platform import gfile
import tensorflow as tf
from onnx_tf.frontend import convert_graph
import onnx
import argparse



def import_file(model_filename,target_node,out_file):
    '''
    Runs import file on the target file name,
    setting the target_node as output
    writing to the designated out_file
    :param model_filename: the absolate path to the import proto file
    :param target_node: the target node on the output
    :param out_file: the output file to write
    :return:
    '''
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
            default_graph = tf.get_default_graph()
            graph_with_output_shapes = default_graph.as_graph_def(add_shapes=True)

            '''
             When loading the graph in to  the tensorflow default graph:
             https://www.tensorflow.org/api_docs/python/tf/get_default_graph

            Tensorflow prepends an import/ namespace to every node.

            '''
            target_node = 'import/' + target_node
            nodes = graph_with_output_shapes.node._values
            for node in nodes:
                if node.name == target_node:
                    out_graph = convert_graph(graph_with_output_shapes, node)
                    out_string = out_graph.SerializeToString()
                    with open(out_file, mode='w+') as f:
                        f.write(out_string)
                    with open(out_file, mode='r') as f:
                        loaded_graph = onnx.load(f)
                        print loaded_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an onnx mapping process on tensorflow pb files.')
    parser.add_argument('--inputfile', dest='input_file',
                        help='the absolute path to the input tensorflow protobuf file')
    parser.add_argument('--targetnode', dest='target_node',
                        help='the target node in the tensorflow protobuf for output')
    parser.add_argument('--outfile', dest='out_file',
                        help='the output file path to write for onnx')

    args = parser.parse_args()
    model_filename = args.input_file
    target_node = args.target_node
    out_file = args.out_file
    import_file(model_filename=model_filename,target_node=target_node,out_file=out_file)