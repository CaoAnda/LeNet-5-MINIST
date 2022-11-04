import argparse  
  
def get_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--half_conv', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--output', action='store_true', default=True)  
  
    opt = parser.parse_args()  
    if opt.output:  
        print(opt)
    return opt  
  
if __name__ == '__main__':  
    opt = get_args()
    # import os
    # if not os.path.exists('./analys.csv'):
    #     with open('./analys.csv', 'w+') as file:
    #         file.write(','.join(['half_conv', 'dropout', 'input_size', 'lr', 'acc']))
    #         file.write('\n')
    # with open('./analys.csv', 'a+') as file:
    #     file.write(','.join(map(str, [False, 0.2, (32, 32)[0], 1e-3, 0.9445])))
    #     file.write('\n')