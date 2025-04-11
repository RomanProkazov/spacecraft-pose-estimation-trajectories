import PySpin

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No camera detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list[0]  # Select first camera
    cam.Init()
    
    cam.BeginAcquisition()
    image_result = cam.GetNextImage()
    print(type(image_result))
    image_result.Save("image.png")

    if image_result.IsIncomplete():
        print(f"Image incomplete with status {image_result.GetImageStatus()}")

    else:
        print("Image captured successfully!")
        width = image_result.GetWidth()
        height = image_result.GetHeight()
        print(f"Image Size: {width}x{height}")

    image_result.Release()
    cam.EndAcquisition()
    cam.DeInit()
    del cam

    cam_list.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
