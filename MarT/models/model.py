from .modeling_unimo import UnimoForMaskedLM
from models.modeling_visual_bert import VisualBertForMaskedLM
from models.vilbert import VilBertForMaskLM
from models.modeling_vilt import ViltForMaskedLM
from models.modeling_flava import FlavaForMaskedLM

class MKGformerKGC(UnimoForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser

class VisualBertKGC(VisualBertForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser

class VilBertKGC(VilBertForMaskLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser
    
class ViltKGC(ViltForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser
    
class FlavaKGC(FlavaForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser
