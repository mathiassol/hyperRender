#pragma once

// Windows' rpcndr.h does `#define hyper __int64`, which breaks `namespace hyper`.
// Pull it in once and take the name back; its include guard keeps it from
// being redefined by anything included later. Every hyper header includes this first.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <rpcndr.h>
#undef hyper
#endif
