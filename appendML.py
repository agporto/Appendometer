import argparse
import utils


ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input-dir",
    type=str,
    default="images",
    help="input directory (default = images)",
    metavar="",
)
ap.add_argument(
    "-p",
    "--predictor",
    type=str,
    default="resources/predictor.dat",
    help="trained shape prediction model (default = resources/predictor.dat)",
    metavar="",
)
ap.add_argument(
    "-o",
    "--out-file",
    type=str,
    default="output.xml",
    help="output file name suffix (default = output.xml)",
    metavar="",
)
ap.add_argument(
    "-l",
    "--ignore-list",
    nargs="*",
    type=int,
    default=None,
    help=" (optional) prevents landmarks of choice from being output",
    metavar="",
)
ap.add_argument(
    "-m",
    "--multiscale",
    type=int,
    default=None,
    help=" performs multiscale prediction to help identify errors (default = None)",
    metavar="",
)


args = vars(ap.parse_args())

utils.predictions_to_xml(
    args["predictor"],
    folder=args["input_dir"],
    ignore=args["ignore_list"],
    output=args["out_file"],
    multiscale=args["multiscale"],

)
if not args["multiscale"]:
    utils.dlib_xml_to_pandas(args["out_file"])
    utils.dlib_xml_to_tps(args["out_file"])
else:
    utils.dlib_xml_to_pandas("high_var_" + args["out_file"])
    utils.dlib_xml_to_tps("high_var_" + args["out_file"])
    utils.dlib_xml_to_pandas("low_var_" + args["out_file"])
    utils.dlib_xml_to_tps("low_var_" + args["out_file"])