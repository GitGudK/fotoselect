"""
Face detection, clustering, and recognition module.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Face detection and recognition
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


class FaceManager:
    """
    Manages face detection, clustering, and recognition.

    Workflow:
    1. Detect faces in photos and extract embeddings
    2. Cluster similar faces together
    3. Allow user to assign names to clusters
    4. Use named clusters to identify people in new photos
    """

    def __init__(
        self,
        data_dir: str = "face_data",
        device: Optional[torch.device] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        print(f"Face recognition using device: {self.device}")

        # Initialize face detector (MTCNN)
        self.detector = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True
        )

        # Initialize face embedding model (FaceNet)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Storage paths
        self.embeddings_path = self.data_dir / "embeddings.pkl"
        self.clusters_path = self.data_dir / "clusters.json"
        self.names_path = self.data_dir / "names.json"

        # Load existing data
        self.face_data = self._load_embeddings()
        self.clusters = self._load_clusters()
        self.cluster_names = self._load_names()

    def _load_embeddings(self) -> Dict:
        """Load saved face embeddings."""
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'rb') as f:
                return pickle.load(f)
        return {"embeddings": [], "face_locations": [], "image_paths": [], "face_ids": []}

    def _save_embeddings(self):
        """Save face embeddings to disk."""
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.face_data, f)

    def _load_clusters(self) -> Dict[int, List[int]]:
        """Load cluster assignments."""
        if self.clusters_path.exists():
            with open(self.clusters_path, 'r') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        return {}

    def _save_clusters(self):
        """Save cluster assignments."""
        with open(self.clusters_path, 'w') as f:
            json.dump(self.clusters, f, indent=2)

    def _load_names(self) -> Dict[int, str]:
        """Load cluster names."""
        if self.names_path.exists():
            with open(self.names_path, 'r') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        return {}

    def _save_names(self):
        """Save cluster names."""
        with open(self.names_path, 'w') as f:
            json.dump(self.cluster_names, f, indent=2)

    @torch.no_grad()
    def detect_faces(self, image_path: str) -> List[Tuple[np.ndarray, np.ndarray, List[int]]]:
        """
        Detect faces in an image and extract embeddings.

        Returns:
            List of (face_image, embedding, bbox) tuples
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return []

        # Detect faces
        boxes, probs = self.detector.detect(img)

        if boxes is None:
            return []

        faces = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.9:  # Confidence threshold
                continue

            # Extract face region
            x1, y1, x2, y2 = [int(b) for b in box]

            # Add margin
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img.width, x2 + margin)
            y2 = min(img.height, y2 + margin)

            face_img = img.crop((x1, y1, x2, y2))
            face_img_resized = face_img.resize((160, 160))

            # Convert to tensor and get embedding
            face_tensor = torch.tensor(np.array(face_img_resized)).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            embedding = self.embedder(face_tensor).cpu().numpy().flatten()

            faces.append((np.array(face_img), embedding, [x1, y1, x2, y2]))

        return faces

    def process_folder(
        self,
        folder: str,
        clear_existing: bool = False
    ) -> int:
        """
        Process all images in a folder and extract face embeddings.

        Args:
            folder: Path to folder containing images
            clear_existing: If True, clear existing face data first

        Returns:
            Number of faces detected
        """
        if clear_existing:
            self.face_data = {"embeddings": [], "face_locations": [], "image_paths": [], "face_ids": []}

        folder_path = Path(folder)
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}

        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))

        image_files = sorted(image_files)
        print(f"Processing {len(image_files)} images...")

        face_count = 0
        face_id = len(self.face_data["embeddings"])

        for img_path in tqdm(image_files, desc="Detecting faces"):
            # Skip if already processed
            if str(img_path) in self.face_data["image_paths"]:
                continue

            faces = self.detect_faces(str(img_path))

            for face_img, embedding, bbox in faces:
                self.face_data["embeddings"].append(embedding)
                self.face_data["face_locations"].append(bbox)
                self.face_data["image_paths"].append(str(img_path))
                self.face_data["face_ids"].append(face_id)
                face_id += 1
                face_count += 1

        self._save_embeddings()
        print(f"Detected {face_count} new faces (total: {len(self.face_data['embeddings'])})")

        return face_count

    def cluster_faces(
        self,
        eps: float = 0.5,
        min_samples: int = 2
    ) -> Dict[int, List[int]]:
        """
        Cluster detected faces using DBSCAN.

        Args:
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples to form a cluster

        Returns:
            Dict mapping cluster_id to list of face_ids
        """
        if len(self.face_data["embeddings"]) == 0:
            print("No faces to cluster")
            return {}

        embeddings = np.array(self.face_data["embeddings"])

        # Use cosine distance for face embeddings
        distances = cosine_distances(embeddings)

        # Cluster with DBSCAN
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distances)

        labels = clustering.labels_

        # Group face IDs by cluster
        clusters = defaultdict(list)
        for face_idx, cluster_id in enumerate(labels):
            face_id = self.face_data["face_ids"][face_idx]
            clusters[int(cluster_id)].append(face_id)

        self.clusters = dict(clusters)
        self._save_clusters()

        # Stats
        n_clusters = len([c for c in self.clusters.keys() if c >= 0])
        n_noise = len(self.clusters.get(-1, []))
        print(f"Found {n_clusters} clusters, {n_noise} unclustered faces")

        return self.clusters

    def set_cluster_name(self, cluster_id: int, name: str):
        """Assign a name to a cluster."""
        self.cluster_names[cluster_id] = name
        self._save_names()
        print(f"Cluster {cluster_id} named: {name}")

    def get_cluster_name(self, cluster_id: int) -> Optional[str]:
        """Get the name assigned to a cluster."""
        return self.cluster_names.get(cluster_id)

    def get_cluster_faces(self, cluster_id: int) -> List[Tuple[str, List[int]]]:
        """
        Get all faces in a cluster.

        Returns:
            List of (image_path, bbox) tuples
        """
        if cluster_id not in self.clusters:
            return []

        faces = []
        for face_id in self.clusters[cluster_id]:
            idx = self.face_data["face_ids"].index(face_id)
            image_path = self.face_data["image_paths"][idx]
            bbox = self.face_data["face_locations"][idx]
            faces.append((image_path, bbox))

        return faces

    def get_cluster_sample_images(
        self,
        cluster_id: int,
        max_samples: int = 5
    ) -> List[np.ndarray]:
        """Get sample face images from a cluster."""
        faces = self.get_cluster_faces(cluster_id)[:max_samples]

        images = []
        for image_path, bbox in faces:
            try:
                img = Image.open(image_path)
                x1, y1, x2, y2 = bbox
                face_img = img.crop((x1, y1, x2, y2))
                images.append(np.array(face_img))
            except Exception as e:
                continue

        return images

    @torch.no_grad()
    def identify_faces(
        self,
        image_path: str,
        threshold: float = 0.6
    ) -> List[Tuple[List[int], str, float]]:
        """
        Identify faces in an image using named clusters.

        Args:
            image_path: Path to image
            threshold: Maximum distance to consider a match

        Returns:
            List of (bbox, name, confidence) tuples
        """
        faces = self.detect_faces(image_path)

        if not faces or not self.cluster_names:
            return []

        # Get mean embedding for each named cluster
        cluster_embeddings = {}
        for cluster_id, name in self.cluster_names.items():
            if cluster_id < 0:
                continue

            face_ids = self.clusters.get(cluster_id, [])
            if not face_ids:
                continue

            embeddings = []
            for face_id in face_ids:
                idx = self.face_data["face_ids"].index(face_id)
                embeddings.append(self.face_data["embeddings"][idx])

            cluster_embeddings[cluster_id] = np.mean(embeddings, axis=0)

        if not cluster_embeddings:
            return []

        # Match detected faces to clusters
        results = []
        for face_img, embedding, bbox in faces:
            best_match = None
            best_distance = float('inf')

            for cluster_id, cluster_emb in cluster_embeddings.items():
                distance = cosine_distances([embedding], [cluster_emb])[0][0]
                if distance < best_distance:
                    best_distance = distance
                    best_match = cluster_id

            if best_match is not None and best_distance < threshold:
                name = self.cluster_names[best_match]
                confidence = 1.0 - best_distance
                results.append((bbox, name, confidence))
            else:
                results.append((bbox, "Unknown", 0.0))

        return results

    def get_photos_by_person(self, name: str) -> List[str]:
        """Get all photos containing a specific person."""
        # Find cluster(s) with this name
        cluster_ids = [cid for cid, n in self.cluster_names.items() if n == name]

        if not cluster_ids:
            return []

        photos = set()
        for cluster_id in cluster_ids:
            for face_id in self.clusters.get(cluster_id, []):
                idx = self.face_data["face_ids"].index(face_id)
                photos.add(self.face_data["image_paths"][idx])

        return sorted(list(photos))

    def get_all_people(self) -> List[Tuple[str, int]]:
        """Get all named people and their photo counts."""
        people = []
        for cluster_id, name in self.cluster_names.items():
            if cluster_id < 0:
                continue
            photo_count = len(set(
                self.face_data["image_paths"][self.face_data["face_ids"].index(fid)]
                for fid in self.clusters.get(cluster_id, [])
            ))
            people.append((name, photo_count))

        return sorted(people, key=lambda x: -x[1])

    def get_summary(self) -> Dict:
        """Get a summary of face data."""
        return {
            "total_faces": len(self.face_data["embeddings"]),
            "total_images": len(set(self.face_data["image_paths"])),
            "total_clusters": len([c for c in self.clusters.keys() if c >= 0]),
            "named_clusters": len(self.cluster_names),
            "unclustered_faces": len(self.clusters.get(-1, []))
        }
