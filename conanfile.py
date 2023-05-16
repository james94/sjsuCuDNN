import distutils.spawn
from conans import ConanFile, CMake, tools

class SjsuCuDNNConan(ConanFile):
    name = "sjsuCuDNN"
    version = "1.0.0"
    license = "MIT"
    author = "James Guzman"
    url = "https://github.com/james94/sjsuCuDNN"
    description = "sjsuCuDNN is an Open Source CUDA library for deep neural networks"
    topics = ("cuda", "sjsuCuDNN", "neural network", "deep learning")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False,
                       "gtest:shared": False}
    generators = "cmake"
    exports_sources = "include/*", "src/*", "tests/*"

    def requirements(self):
        # self.requires("cuda/11.7")
        self.requires("gtest/1.11.0")

    def build(self):
        cmake = CMake(self)
        # cmake.definitions["ENABLE_TESTS"] = self.options.build_tests
        # cmake.definitions["CMAKE_TOOLCHAIN_FILE"] = "conan_paths.cmake" # Specify the Conan toolchain file 
        cmake.configure(source_folder="src")
        cmake.build()
        cmake.install()

    def package(self):
        self.copy("*.h", dst="include/sjsuCuDNN", src="include/sjsuCuDNN")
        self.copy("*.cuh", dst="include/sjsuCuDNN", src="include/sjsuCuDNN")
        self.copy("*.a", dst="lib", keep_path=False)
        # self.copy("*.so", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["sjsuCuDNN"]

    def get_cuda_path(self):
        nvcc_path = distutils.spawn.find_executable("nvcc")
        if nvcc_path:
            cuda_path = nvcc_path[:-len("/bin/nvcc")]
            return cuda_path
        else:
            print("CUDA path not found")
        content = """\
            set(CONAN_CUDA_PATH "{cuda_path}")
            """.format(cuda_path=cuda_path)
        return content

    # def generate(self):
    #     # Generate a Conan paths file to set the CUDA environment variables
    #     self.output.info("Generating conan_paths.cmake")
    #     content = tools.load("conan_paths.cmake")
    #     tools.save("conan_paths.cmake", content.replace(self.get_cuda_path(), self.deps_cpp_info["cuda"].rootpath))
