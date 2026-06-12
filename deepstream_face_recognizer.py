import os
import platform
import threading
import traceback

import cv2
import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject, GLib

try:
    import pyds
except ImportError as exc:
    raise ImportError(
        "Failed to import pyds. NVIDIA DeepStream Python bindings must be installed. "
        "Run this app from an environment where DeepStream is available."
    ) from exc


class DeepStreamFaceRecognizer:
    def __init__(
        self,
        detector_config: str,
        face_embedder,
        face_db,
        camera_index: int = 0,
        preview_width: int = 1280,
        preview_height: int = 720,
        max_track_distance: float = 120.0,
        max_missed_frames: int = 15,
    ):
        self.detector_config = detector_config
        self.face_embedder = face_embedder
        self.face_db = face_db
        self.camera_index = camera_index
        self.preview_width = preview_width
        self.preview_height = preview_height

        self.tracks = {}
        self.next_track_id = 1
        self.max_track_distance = max_track_distance
        self.max_missed_frames = max_missed_frames

        Gst.init(None)
        self.pipeline = self._build_pipeline()
        self.main_loop = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("deepstream-face-recognition")
        if not pipeline:
            raise RuntimeError("Unable to create DeepStream pipeline")

        if platform.system() == "Windows":
            source = Gst.ElementFactory.make("ksvideosrc", "camera-source")
            source.set_property("device-index", self.camera_index)
        else:
            source = Gst.ElementFactory.make("v4l2src", "camera-source")
            source.set_property("device", f"/dev/video{self.camera_index}")

        if not source:
            raise RuntimeError("Could not create camera source element")

        convert_src = Gst.ElementFactory.make("videoconvert", "convert-source")
        capsfilter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        caps = Gst.Caps.from_string(
            f"video/x-raw, width={self.preview_width}, height={self.preview_height}, framerate=30/1"
        )
        capsfilter.set_property("caps", caps)

        tee = Gst.ElementFactory.make("tee", "tee")
        queue_infer = Gst.ElementFactory.make("queue", "queue-infer")
        queue_cpu = Gst.ElementFactory.make("queue", "queue-cpu")

        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
        caps_nv12 = Gst.ElementFactory.make("capsfilter", "capsfilter-nv12")
        caps_nv12.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"),
        )

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property("width", self.preview_width)
        streammux.set_property("height", self.preview_height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)

        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", self.detector_config)

        tracker = None
        if self.enable_nvtracker:
            tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if not tracker:
                raise RuntimeError("Could not create nvtracker element. Ensure DeepStream nvtracker is available.")
            tracker.set_property("config-file-path", self.tracker_config)

        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
        osd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
        sink = Gst.ElementFactory.make("nveglglessink", "video-renderer")
        sink.set_property("sync", False)
        sink.set_property("qos", False)

        convert_cpu = Gst.ElementFactory.make("videoconvert", "convert-cpu")
        caps_bgr = Gst.ElementFactory.make("capsfilter", "capsfilter-bgr")
        caps_bgr.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format=BGRx, width={self.preview_width}, height={self.preview_height}"
            ),
        )
        appsink = Gst.ElementFactory.make("appsink", "appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("caps", caps_bgr.get_property("caps"))

        elements = [source, convert_src, capsfilter, tee, queue_infer, queue_cpu, nvvidconv, caps_nv12, streammux, pgie]
        if tracker is not None:
            elements.append(tracker)
        elements.extend([nvvidconv2, osd, sink, convert_cpu, caps_bgr, appsink])
        for element in elements:
            if not element:
                raise RuntimeError("Failed to create one of the required GStreamer elements")
            pipeline.add(element)

        if not Gst.Element.link(source, convert_src):
            raise RuntimeError("Failed to link source -> convert")
        if not Gst.Element.link(convert_src, capsfilter):
            raise RuntimeError("Failed to link convert -> capsfilter")
        if not Gst.Element.link(capsfilter, tee):
            raise RuntimeError("Failed to link capsfilter -> tee")

        if not Gst.Element.link(tee, queue_infer):
            raise RuntimeError("Failed to link tee -> queue_infer")
        if not Gst.Element.link(queue_infer, nvvidconv):
            raise RuntimeError("Failed to link queue_infer -> nvvidconv")
        if not Gst.Element.link(nvvidconv, caps_nv12):
            raise RuntimeError("Failed to link nvvidconv -> caps_nv12")

        srcpad = caps_nv12.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Failed to get caps_nv12 src pad")
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            raise RuntimeError("Failed to request streammux sink pad")
        if Gst.Pad.link(srcpad, sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_nv12 -> streammux")

        if not Gst.Element.link(streammux, pgie):
            raise RuntimeError("Failed to link streammux -> pgie")
        if tracker is not None:
            if not Gst.Element.link(pgie, tracker):
                raise RuntimeError("Failed to link pgie -> tracker")
            if not Gst.Element.link(tracker, nvvidconv2):
                raise RuntimeError("Failed to link tracker -> nvvidconv2")
        else:
            if not Gst.Element.link(pgie, nvvidconv2):
                raise RuntimeError("Failed to link pgie -> nvvidconv2")
        if not Gst.Element.link(nvvidconv2, osd):
            raise RuntimeError("Failed to link nvvidconv2 -> osd")
        if not Gst.Element.link(osd, sink):
            raise RuntimeError("Failed to link osd -> sink")

        if not Gst.Element.link(tee, queue_cpu):
            raise RuntimeError("Failed to link tee -> queue_cpu")
        if not Gst.Element.link(queue_cpu, convert_cpu):
            raise RuntimeError("Failed to link queue_cpu -> convert_cpu")
        if not Gst.Element.link(convert_cpu, appsink):
            raise RuntimeError("Failed to link convert_cpu -> appsink")

        appsink.connect("new-sample", self._on_new_sample)

        osd_sink_pad = osd.get_static_pad("sink")
        if not osd_sink_pad:
            raise RuntimeError("Unable to get snapshot pad from OSD")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_buffer_probe, None)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        return pipeline

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            raw = np.frombuffer(map_info.data, np.uint8)
            raw = raw.reshape((height, width, 4))
            frame = raw[:, :, :3].copy()
            with self.frame_lock:
                self.latest_frame = frame
        except Exception:
            pass
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _osd_buffer_probe(self, pad, info, u_data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_meta = pyds.glist_first(batch_meta.frame_meta_list)
        while frame_meta:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_meta.data)
            frame = self._get_latest_frame()
            if frame is None:
                frame_meta = pyds.glist_next(frame_meta)
                continue

            detections = []
            obj_meta_ptr = pyds.glist_first(frame_meta.obj_meta_list)
            while obj_meta_ptr:
                obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_ptr.data)
                left = int(max(obj_meta.rect_params.left, 0))
                top = int(max(obj_meta.rect_params.top, 0))
                width = int(obj_meta.rect_params.width)
                height = int(obj_meta.rect_params.height)
                right = min(left + width, frame.shape[1])
                bottom = min(top + height, frame.shape[0])

                if right > left and bottom > top:
                    track_id = None
                    if self.enable_nvtracker and getattr(obj_meta, "object_id", None) is not None:
                        try:
                            track_id = int(obj_meta.object_id)
                        except Exception:
                            track_id = None

                    detections.append(
                        {
                            "obj_meta": obj_meta,
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "track_id": track_id,
                        }
                    )

                obj_meta_ptr = pyds.glist_next(obj_meta_ptr)

            if not self.enable_nvtracker:
                self._assign_track_ids(detections)
            for detection in detections:
                self._process_detection(
                    detection["obj_meta"],
                    frame,
                    track_id=detection.get("track_id"),
                )

            frame_meta = pyds.glist_next(frame_meta)

        return Gst.PadProbeReturn.OK

    def _process_detection(self, obj_meta, frame, track_id=None):
        left = int(max(obj_meta.rect_params.left, 0))
        top = int(max(obj_meta.rect_params.top, 0))
        width = int(obj_meta.rect_params.width)
        height = int(obj_meta.rect_params.height)
        right = min(left + width, frame.shape[1])
        bottom = min(top + height, frame.shape[0])

        if right <= left or bottom <= top:
            return

        face_image = frame[top:bottom, left:right]
        if face_image.size == 0:
            return

        try:
            embedding = self.face_embedder.encode(face_image)
        except Exception:
            return

        person_id, distance = self.face_db.identify(embedding)
        if person_id is None:
            person_id = self.face_db.create_person(embedding)
            label = f"New face {person_id}"
        else:
            self.face_db.update_person(person_id, embedding)
            label = f"Person {person_id}"

        if track_id is not None:
            label = f"T{track_id}: {label}"
            try:
                obj_meta.object_id = int(track_id)
            except Exception:
                pass

        self._annotate_object(obj_meta, label, left, top)

    def _annotate_object(self, obj_meta, label: str, left: int, top: int):
        try:
            obj_meta.text_params.display_text = label
            obj_meta.text_params.x_offset = left
            obj_meta.text_params.y_offset = max(top - 12, 0)
            obj_meta.text_params.font_params.font_size = 14
            self._set_color(obj_meta.text_params.font_params.font_color, (1.0, 1.0, 0.0, 1.0))
            obj_meta.text_params.set_bg_clr = 1
            self._set_color(obj_meta.text_params.text_bg_clr, (0.0, 0.0, 0.0, 0.7))
            self._set_color(obj_meta.rect_params.border_color, (0.0, 1.0, 0.0, 1.0))
        except Exception:
            pass

    def _assign_track_ids(self, detections):
        if len(self.tracks) == 0:
            for detection in detections:
                detection["track_id"] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    "left": detection["left"],
                    "top": detection["top"],
                    "right": detection["right"],
                    "bottom": detection["bottom"],
                    "missed": 0,
                }
                self.next_track_id += 1
            return

        assigned = set()
        for detection in detections:
            best_track_id = None
            best_distance = float("inf")
            cx = (detection["left"] + detection["right"]) / 2.0
            cy = (detection["top"] + detection["bottom"]) / 2.0
            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue
                tx = (track["left"] + track["right"]) / 2.0
                ty = (track["top"] + track["bottom"]) / 2.0
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None and best_distance <= self.max_track_distance:
                detection["track_id"] = best_track_id
                assigned.add(best_track_id)
                self.tracks[best_track_id].update(
                    {
                        "left": detection["left"],
                        "top": detection["top"],
                        "right": detection["right"],
                        "bottom": detection["bottom"],
                        "missed": 0,
                    }
                )
            else:
                detection["track_id"] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    "left": detection["left"],
                    "top": detection["top"],
                    "right": detection["right"],
                    "bottom": detection["bottom"],
                    "missed": 0,
                }
                self.next_track_id += 1

        for track_id in list(self.tracks.keys()):
            if track_id not in assigned:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

    def _set_color(self, color_obj, rgba):
        try:
            color_obj.set(*rgba)
        except Exception:
            try:
                color_obj.red, color_obj.green, color_obj.blue, color_obj.alpha = rgba
            except Exception:
                pass

    def _get_latest_frame(self):
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def _on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"DeepStream pipeline error: {err.message}")
            print(debug)
            if self.main_loop:
                self.main_loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print("End-Of-Stream reached")
            if self.main_loop:
                self.main_loop.quit()

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self.main_loop = GLib.MainLoop()
        try:
            print("Starting DeepStream face recognition pipeline...")
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
