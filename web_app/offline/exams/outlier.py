import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OutlierDetector:
    def __init__(self, data):
        self.data = self.standard_scaler(data)
        self.q1 = np.percentile(self.data, 25)
        self.q3 = np.percentile(self.data, 75)
        self.iqr = self.q3 - self.q1

    def standard_scaler(self, data):
        scaler = StandardScaler()
        flow_array = np.array(data).reshape(-1, 1)
        return scaler.fit_transform(flow_array)

    def iqr_contamination(self, multiplier=1.5):
        """IQR method to estimate contamination"""
        lower_bound = self.q1 - multiplier * self.iqr
        upper_bound = self.q3 + multiplier * self.iqr
        iqr_anomalies = (self.data < lower_bound) | (self.data > upper_bound)
        contamination_estimate = np.sum(iqr_anomalies) / len(self.data)
        return contamination_estimate, iqr_anomalies

    def isolation_forest_outliers(self, contamination):
        """Use Isolation Forest to detect outliers based on contamination estimate"""
        iso_forest = IsolationForest(
            contamination=contamination, n_estimators=200, random_state=42
        )
        iso_forest.fit(self.data)
        predictions = iso_forest.predict(self.data)
        # -1 indicates an anomaly, 1 indicates a normal data point
        anomalies = predictions == -1
        anomalies_idx = np.where(predictions == -1)[0]
        return anomalies, anomalies_idx

    def detect_outliers(self):
        """Combine both methods: estimate contamination and use Isolation Forest"""
        iqr_contamination_level, iqr_anomalies = self.iqr_contamination()

        # Use Isolation Forest with the estimated contamination level
        outliers, anomalies_idx = self.isolation_forest_outliers(contamination=iqr_contamination_level)

        n_outliers = np.sum(outliers)
        n_iqr_outliers = np.sum(iqr_anomalies)
        avg_n_ouliers = (n_iqr_outliers + n_outliers) / 2
        contamination_level = f"{iqr_contamination_level:.2%}"

        anomaly = round((100 * (n_outliers / len(self.data))), 3)
        percent_anomaly = f"{anomaly}%"

        return (
            dict(
                mean_outliers=avg_n_ouliers,
                contamination_level=contamination_level,
                n_iqr_anomalies=n_iqr_outliers,
                n_anomalies=n_outliers,
                score=percent_anomaly,
            ),
            anomalies_idx,
            anomaly
        )


# Example Usage
if __name__ == "__main__":
    ...
    # Initialize the detector
    # detector = CombinedOutlierDetector(data)

    # Detect outliers
    # outliers = detector.detect_outliers()

    # print("Outliers: ", outliers)
