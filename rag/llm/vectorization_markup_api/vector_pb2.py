# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: vector.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'vector.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cvector.proto\x12\x06vector\"\x16\n\x05Texts\x12\r\n\x05texts\x18\x01 \x03(\t\"\x1a\n\nTextVector\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\"+\n\x04\x44\x61ta\x12#\n\x07vectors\x18\x01 \x03(\x0b\x32\x12.vector.TextVector24\n\x06Vector\x12*\n\tGetVector\x12\r.vector.Texts\x1a\x0c.vector.Data\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vector_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TEXTS']._serialized_start=24
  _globals['_TEXTS']._serialized_end=46
  _globals['_TEXTVECTOR']._serialized_start=48
  _globals['_TEXTVECTOR']._serialized_end=74
  _globals['_DATA']._serialized_start=76
  _globals['_DATA']._serialized_end=119
  _globals['_VECTOR']._serialized_start=121
  _globals['_VECTOR']._serialized_end=173
# @@protoc_insertion_point(module_scope)
