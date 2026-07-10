include(CheckCSourceCompiles)

set(HAVE_AVX2 FALSE)
set(HAVE_AVX512 FALSE)
set(HAVE_NEON FALSE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i686|i386)")
  check_c_source_compiles("
    #include <immintrin.h>
    int main() {
      __m256i a = _mm256_setzero_si256();
      __m256i b = _mm256_shuffle_epi8(a, a);
      (void)b;
      return 0;
    }
  " HAVE_AVX2)
  if(HAVE_AVX2)
    set(OIL_AVX2 ON CACHE INTERNAL "AVX2 support")
    message(STATUS "ARCH: AVX2 detected")
  endif()

  check_c_source_compiles("
    #include <immintrin.h>
    int main() {
      __m512i a = _mm512_setzero_si512();
      (void)a;
      return 0;
    }
  " HAVE_AVX512)
  if(HAVE_AVX512)
    set(OIL_AVX512 ON CACHE INTERNAL "AVX-512 support")
    message(STATUS "ARCH: AVX-512 detected")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)")
  check_c_source_compiles("
    #include <arm_neon.h>
    int main() {
      int8x16_t a = vdupq_n_s8(0);
      (void)a;
      return 0;
    }
  " HAVE_NEON)
  if(HAVE_NEON)
    set(OIL_NEON ON CACHE INTERNAL "NEON support")
    message(STATUS "ARCH: NEON detected")
  endif()
endif()
