import argparse
import glob
import os
import boto3
from time import time
from distutils.util import strtobool

import filetype
from tqdm import tqdm

from ..bg import remove




def main():
    model_path = os.environ.get(
        "U2NETP_PATH",
        os.path.expanduser(os.path.join("~", ".u2net")),
    )

    model_choices = [os.path.splitext(os.path.basename(x))[0] for x in set(glob.glob(model_path + "/*"))]
    if len(model_choices) == 0:
        model_choices = ["u2net", "u2netp", "u2net_human_seg"]

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-b",
        "--bucket",
        default="yogupta",
        type=str,
        help="bucket name",
    )

    ap.add_argument(
        "-m",
        "--model",
        default="u2net",
        type=str,
        choices=model_choices,
        help="The model name.",
    )

    ap.add_argument(
        "-a",
        "--alpha-matting",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="When true use alpha matting cutout.",
    )

    ap.add_argument(
        "-af",
        "--alpha-matting-foreground-threshold",
        default=240,
        type=int,
        help="The trimap foreground threshold.",
    )

    ap.add_argument(
        "-ab",
        "--alpha-matting-background-threshold",
        default=10,
        type=int,
        help="The trimap background threshold.",
    )

    ap.add_argument(
        "-ae",
        "--alpha-matting-erode-size",
        default=10,
        type=int,
        help="Size of element used for the erosion.",
    )

    ap.add_argument(
        "-az",
        "--alpha-matting-base-size",
        default=1000,
        type=int,
        help="The image base size.",
    )

    ap.add_argument(
        "-p",
        "--path",
        nargs=2,
        help="An input folder and an output folder.",
    )
    ap.add_argument(
        "--s3",
        default="",
        type=bool,
        help="use s3 bucket or not",
    )
    ap.add_argument(
        "--s3_key_id",
        default="",
        type=str,
    )

    ap.add_argument(
        "--s3_key",
        default="",
        type=str,
    )

    ap.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="-",
        type=argparse.FileType("wb"),
        help="Path to the output png image.",
    )


    ap.add_argument(
        "input",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to the input image.",
    )

    args = ap.parse_args()

    s3 = boto3.client('s3', aws_access_key_id=args.s3_key_id, aws_secret_access_key=args.s3_key)

    r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
    w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)

    if args.path:
        full_paths = [os.path.abspath(path) for path in args.path]

        if args.s3:
            s3_inputs = s3.list_objects_v2(Bucket=args.bucket)
            contents = s3_inputs["Contents"]
            for obj in contents:
                if str(obj["Key"]).startswith("input") and str(obj["Key"]) != "input/":
                    print("filename:", obj["Key"])
                    s3.download_file(args.bucket, obj["Key"], obj["Key"])

        input_paths = [full_paths[0]]
        output_path = full_paths[1]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = set()

        for path in input_paths:
            if os.path.isfile(path):
                files.add(path)
            else:
                input_paths += set(glob.glob(path + "/*"))

        for fi in tqdm(files):
            fi_type = filetype.guess(fi)

            if fi_type is None:
                continue
            elif fi_type.mime.find("image") < 0:
                continue

            with open(fi, "rb") as input:
                with open(os.path.join(output_path, os.path.splitext(os.path.basename(fi))[0] + ".png"), "wb") as output:
                    w(
                        output,
                        remove(
                            r(input),
                            model_name=args.model,
                            alpha_matting=args.alpha_matting,
                            alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                            alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                            alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                            alpha_matting_base_size=args.alpha_matting_base_size,
                        ),
                    )
                    if args.s3:
                        print(os.path.basename(output.name), "here")
                        s3.upload_file(output.name, args.bucket, "output/"+os.path.basename(output.name))

    else:
        w(
            args.output,
            remove(
                r(args.input),
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ),
        )


if __name__ == "__main__":
    main()
