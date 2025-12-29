import { useState, useEffect } from "react";

export default function Recall() {
  const [formData, setFormData] = useState({
    name: "America's joy",
    form: "UR",
    rawErg: "69",
    erg: "72",
    yob: "2018",
    sex: "F",
    sire: "American pharoah",
    fee: "100000",
    crop: "2",
    dam: "Leslie's Lady",
    damForm: "LRw",
    ems3: "61",
    bmSire: "Tricky Creek",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || "";

  // Helper function to check inputs before sending
  const safeNumber = (value, type = "float") => {
    if (value === "" || value === null || value === undefined) return 0;
    const n = type === "int" ? parseInt(value) : parseFloat(value);
    return isNaN(n) ? null : n;
  };

  // Helper function ot check string inputs before sending
  const safeString = (value) => {
    if (!value || value.trim() === "") {
      return "";
    }
    return value.trim();
  };

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(errorText);
        }

        const data = await res.json();
        setBackendStatus(data);
      } catch (err) {
        console.error("Health check failed:", err.message);
        setBackendStatus({ status: "offline" });
      }
    };
    checkHealth();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const payload = {
        name: safeString(formData.name),
        form: safeString(formData.form),
        rawErg: safeNumber(formData.rawErg),
        erg: safeNumber(formData.erg),
        yob: safeNumber(formData.yob),
        sex: formData.sex,
        sire: safeString(formData.sire),
        fee: safeNumber(formData.fee),
        crop: safeNumber(formData.crop),
        dam: safeString(formData.dam),
        damForm: safeString(formData.damForm),
        ems3: safeNumber(formData.ems3),
        bmSire: safeString(formData.bmSire),
      };
      console.log(payload, typeof payload);

      const response = await fetch(`${API_URL}/recall`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let errText = "Prediction failed";
        try {
          const errorBody = await response.json();
          if (errorBody?.detail) {
            errText =
              typeof errorBody.detail === "string"
                ? errorBody.detail
                : JSON.stringify(errorBody.detail);
          } else if (errorBody?.message) {
            errText = errorBody.message;
          } else {
            errText = JSON.stringify(errorBody);
          }
        } catch (e) {
          // not JSON, try text
          try {
            errText = await response.text();
          } catch {
            /* keep fallback */
          }
        }
        throw new Error(errText);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(
        err.message ||
          "Failed to get prediction. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-900 via-gray-900 to-slate-950 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-zinc-800 backdrop-blur rounded-2xl shadow-2xl p-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-grey-800 to-slate-850 bg-clip-text text-transparent">
                Recall Horse Rating
              </h1>
            </div>

            {backendStatus && (
              <div
                className={`px-4 py-2 rounded-full text-sm font-medium ${
                  backendStatus.status === "healthy"
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {backendStatus.status === "healthy" ? "● Online" : "● Offline"}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <div className="lg:col-span-2">
              <h2 className="text-xl font-semibold text-white/80 mb-4">
                Horse Information
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Horse Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    placeholder="Enter horse name"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Sex
                  </label>
                  <select
                    name="sex"
                    value={formData.sex}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="M">Male (M)</option>
                    <option value="F">Female (F)</option>
                    <option value="G">Gelding (G)</option>
                    <option value="C">Colt (C)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Year of Birth
                  </label>
                  <input
                    type="number"
                    name="yob"
                    value={formData.yob}
                    onChange={handleChange}
                    required
                    placeholder="e.g., 2020"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Form
                  </label>
                  <input
                    type="text"
                    name="form"
                    value={formData.form}
                    onChange={handleChange}
                    placeholder="e.g., LR"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Dam Form
                  </label>
                  <input
                    type="text"
                    name="damForm"
                    value={formData.damForm}
                    onChange={handleChange}
                    placeholder="Mother's form"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Raw ERG
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    name="rawErg"
                    value={formData.rawErg}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    ERG
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    name="erg"
                    value={formData.erg}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    EMS3
                  </label>
                  <input
                    type="number"
                    name="ems3"
                    value={formData.ems3}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Crop
                  </label>
                  <input
                    type="number"
                    name="crop"
                    value={formData.crop}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-white/80 mb-4">
                Pedigree
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Sire (Father)
                  </label>
                  <input
                    type="text"
                    name="sire"
                    value={formData.sire}
                    onChange={handleChange}
                    placeholder="Father's name"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Dam (Mother)
                  </label>
                  <input
                    type="text"
                    name="dam"
                    value={formData.dam}
                    onChange={handleChange}
                    placeholder="Mother's name"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Broodmare Sire
                  </label>
                  <input
                    type="text"
                    name="bmSire"
                    value={formData.bmSire}
                    onChange={handleChange}
                    placeholder="Maternal grandfather"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/80 mb-1">
                    Stud Fee
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    name="fee"
                    value={formData.fee}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading || backendStatus?.status !== "healthy"}
            className="w-full bg-gradient-to-r from-neutral-900 via-gray-800 to-slate-800 text-white py-4 rounded-lg font-semibold text-lg hover:from-gray-900 hover:to-slate-900 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
          >
            {loading ? "Predicting..." : " Predict Horse Rating"}
          </button>

          {error && (
            <div className="mt-6 bg-red-50 border-l-4 border-red-500 rounded-lg p-4">
              <div className="flex items-start">
                <span className="text-2xl mr-3">⚠️</span>
                <div>
                  <p className="font-semibold text-red-800">Error</p>
                  <p className="text-red-700 text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {prediction && (
            <div className="mt-6 bg-gradient-to-r from-green-50 to-emerald-50 border-l-4 border-green-500 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-green-900 mb-1">
                    Prediction Complete
                  </h2>
                  <p className="text-green-700">
                    AI model has analyzed the horse data
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-green-600 font-medium">
                    Predicted Rating
                  </p>
                  <p className="text-5xl font-bold text-green-700">
                    {prediction.predicted_rating.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
