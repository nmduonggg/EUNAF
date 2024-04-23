import os

def set_template(args):
    if args.template == 'SuperNet_udl':
        print('[INFO] Template found (UDL-like SR)')
        args.style='Y'
        args.rgb_range=1.0
        args.core='SuperNet_udl'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif args.template == 'SuperNet_udl_RGB':
        print('[INFO] Template found (UDL-like SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='SuperNet_udl'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif  args.template == 'SuperNet_separate':
        print('[INFO] Template found (Separate SR)')
        args.style='Y'
        args.rgb_range=1.0
        args.core='SuperNet_separate'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    elif  args.template == 'SuperNet_separate_RGB':
        print('[INFO] Template found (Separate SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='SuperNet_separate'
        args.weight = os.path.join(args.cv_dir, '_best.t7')
    else:
        print('[ERRO] Template not found')
        assert(0)
