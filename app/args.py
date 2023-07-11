import argparse

parser = argparse.ArgumentParser(
            prog='autoencoder.py',
            usage='Please input some parameters',
            description='description',
            epilog='end',
            add_help=True
            )

# 中間層の次元数
parser.add_argument('-d', '--dimention', default=2, required=False, type=int)
# バッチサイズ
parser.add_argument('-b', '--batchsize', default=256, required=False, type=int)
# エポック数
parser.add_argument('-e', '--epochs', default=20, required=False, type=int)

args = parser.parse_args()
dimention = args.dimention
batchsize = args.batchsize
epochs = args.epochs

print('d: {}'.format(args.dimention))
print('b: {}'.format(args.batchsize))
print('e: {}'.format(args.epochs))