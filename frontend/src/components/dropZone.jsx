import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";

function DropZone(props) {
  const [imageUrl, setImageUrl] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const formData = new FormData();
    formData.append("image", acceptedFiles[0]);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/enhance_image",
        formData,
        {
          responseType: "arraybuffer",
        }
      );
      const data = response.data;
      const blob = new Blob([data]);
      const imageUrl = URL.createObjectURL(blob);
      console.log(data, blob, imageUrl);
      setImageUrl(imageUrl);
    } catch (error) {
      console.error("Error uploading file", error);
    }
  }, []);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
    acceptedFiles,
    fileRejections,
  } = useDropzone({
    onDrop,
    maxFiles: 1,
    accept: {
      "image/jpeg": [],
      "image/png": [],
    },
  });

  const acceptedFileItems = acceptedFiles.map((file) => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
    </li>
  ));

  const fileRejectionItems = fileRejections.map(({ file, errors }) => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
      <ul>
        {errors.map((e) => (
          <li key={e.code}>{e.message}</li>
        ))}
      </ul>
    </li>
  ));

  return (
    <section className="justify-center flex flex-col items-center h-screen w-screen">
      {imageUrl ? (
        <div className="mt-4">
          <img className="rounded-xl m-5" src={imageUrl} alt="Enhanced Image" />
        </div>
      ) : (
        <div
          className="flex flex-col justify-center h-5/6 rounded-xl bg-gray-200 aspect-square "
          {...getRootProps({})}
        >
          <input {...getInputProps()} />
          <div className="flex items-center justify-center outline-dashed outline-2 outline-black h-2/3 m-auto w-2/3 rounded-xl text-lg">
            {isDragAccept && <p>All files will be accepted</p>}
            {isDragReject && <p>Some file formats are not supported</p>}
            {!isDragActive && (
              <p className="cursor-pointer">Drop some files here...</p>
            )}
          </div>
          <em className="">(max number of files: 1)</em>
        </div>
      )}
      <div>
        <div className="flex gap-2">
          <h4>Accepted files:</h4>
          <ul className="flex gap-2">{acceptedFileItems}</ul>
        </div>
        <hr className="bg-black p-[0.5px]" />
        <div className="flex gap-2">
          <h4>Rejected files: </h4>
          <ul>{fileRejectionItems}</ul>
        </div>
      </div>
    </section>
  );
}

export default DropZone;
