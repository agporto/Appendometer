import argparse
import utils
import ntpath

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
    "--max-error",
    type=int,
    default=None,
    help=" maximum prediction error (default = None)",
    metavar="",
)


args = vars(ap.parse_args())

utils.predictions_to_xml(
    args["predictor"],
    folder=args["input_dir"],
    ignore=args["ignore_list"],
    output=args["out_file"],
    max_error=args["max_error"],

)
if args["max_error"] is None:
    utils.dlib_xml_to_pandas(args["out_file"])
    utils.dlib_xml_to_tps(args["out_file"])
else:
    # check if the file exists prior to converting to pandas and tps
    if ntpath.exists("error_" + args["out_file"]):
        utils.dlib_xml_to_pandas("error_" + args["out_file"])
        utils.dlib_xml_to_tps("error_" + args["out_file"])
    if ntpath.exists("accurate_" + args["out_file"]):
        utils.dlib_xml_to_pandas("accurate_" + args["out_file"])
        utils.dlib_xml_to_tps("accurate_" + args["out_file"])
