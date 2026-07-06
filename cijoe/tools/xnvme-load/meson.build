project('xnvme-load', 'c',
  version: '0.1.0',
  license: 'BSD-3-Clause',
  meson_version: '>=0.59',
  default_options: [
    'c_std=gnu11',
    'warning_level=2',
    'buildtype=release',
  ],
)

xnvme_dep = dependency('xnvme', version: '>=0.7.0')
uring_dep = dependency('liburing')

cc = meson.get_compiler('c')
cudart_dep = cc.find_library('cudart',
  dirs: ['/usr/local/cuda/lib64'],
  required: true,
)
cuda_inc = include_directories('/usr/local/cuda/include')

executable('xnvme-load',
  'xnvme_load.c',
  dependencies: [xnvme_dep, cudart_dep],
  include_directories: [cuda_inc],
  c_args: ['-DWITH_CUDA=1'],
  install: true,
)

executable('io-uring-load',
  'io_uring_load.c',
  dependencies: [uring_dep, cudart_dep],
  include_directories: [cuda_inc],
  c_args: ['-DWITH_CUDA=1'],
  install: true,
)

executable('xal-load',
  'xal_load.c',
  dependencies: [xnvme_dep, cudart_dep],
  include_directories: [cuda_inc],
  c_args: ['-DWITH_CUDA=1'],
  install: true,
)

executable('iou-ring-load',
  'iou_ring_load.c',
  dependencies: [uring_dep, cudart_dep],
  include_directories: [cuda_inc],
  c_args: ['-DWITH_CUDA=1'],
  install: true,
)
