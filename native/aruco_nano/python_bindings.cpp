
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>

#include "aruco_nano.h"

#include <stdexcept>
#include <vector>

namespace py = pybind11;

using ArucoNanoParameters = aruco_nano::DetectorParameters;

class PyArucoNanoDetector
{
  public:
    explicit PyArucoNanoDetector(
        int dictionary_id,
        ArucoNanoParameters params = {})
    {
        auto dictionary = cv::aruco::getPredefinedDictionary(dictionary_id);

        // The upstream defaults may contain another dictionary.
        // Let ArucoDetector append only the requested dictionary.
        params.dicts.clear();

        detector_ = aruco_nano::ArucoDetector(dictionary, params);
    }

    py::tuple detect(py::array image)
    {
        auto info = image.request();

        if (info.ndim != 2 && info.ndim != 3)
        {
            throw std::runtime_error(
                "Image must be grayscale or a color image.");
        }

        if (info.shape[0] <= 0 || info.shape[1] <= 0)
        {
            throw std::runtime_error("Image must not be empty.");
        }

        if (info.format != py::format_descriptor<unsigned char>::format())
        {
            throw std::runtime_error("Image must have uint8 dtype.");
        }

        if (!(image.flags() & py::array::c_style))
        {
            throw std::runtime_error(
                "Image must be contiguous. "
                "Use np.ascontiguousarray(frame).");
        }

        int type;

        if (info.ndim == 2)
        {
            type = CV_8UC1;
        }
        else
        {
            if (info.shape[2] != 3)
            {
                throw std::runtime_error(
                    "Color images must have 3 channels (BGR).");
            }

            type = CV_8UC3;
        }

        // This cv::Mat references NumPy's memory; it does not copy pixels.
        cv::Mat frame(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            type,
            info.ptr);

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;

        {
            py::gil_scoped_release release;
            detector_.detectMarkers(frame, corners, ids);
        }

        const py::ssize_t count = static_cast<py::ssize_t>(ids.size());

        py::array_t<float> corner_array(
            py::array::ShapeContainer{
                count, py::ssize_t(4), py::ssize_t(2)});

        py::array_t<int> id_array(
            py::array::ShapeContainer{count});

        auto c = corner_array.mutable_unchecked<3>();
        auto id = id_array.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < count; ++i)
        {
            id(i) = ids[i];

            for (py::ssize_t j = 0; j < 4; ++j)
            {
                c(i, j, 0) = corners[i][j].x;
                c(i, j, 1) = corners[i][j].y;
            }
        }

        return py::make_tuple(corner_array, id_array);
    }

  private:
    aruco_nano::ArucoDetector detector_;
};

PYBIND11_MODULE(aruco_nano_ext, m)
{
    m.doc() = "Python bindings for ArucoNano marker detection";

    py::class_<ArucoNanoParameters>(m, "DetectorParameters")
        .def(py::init<>())

        .def_readwrite(
            "error_correction_rate",
            &ArucoNanoParameters::errorCorrectionRate)

        .def_readwrite(
            "min_size",
            &ArucoNanoParameters::minSize)

        .def_readwrite(
            "max_attempts_per_candidate",
            &ArucoNanoParameters::maxAttemptsPerCandidate)

        .def_readwrite(
            "detect_inverted_marker",
            &ArucoNanoParameters::detectInvertedMarker);

    py::class_<PyArucoNanoDetector>(m, "Detector")
        .def(
            py::init<int, ArucoNanoParameters>(),
            py::arg("dictionary_id") = 2,
            py::arg("params") = ArucoNanoParameters{})
        .def("detect", &PyArucoNanoDetector::detect);
}
