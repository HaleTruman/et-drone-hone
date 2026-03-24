// Status: in development.
// Provides native runtime sensor behavior for BP_DroneSensors:
// - config apply validation
// - viewpoint snapshot query
// - runtime scene-capture PNG readback

#pragma once

#include "Components/ActorComponent.h"
#include "DroneSensorsRuntimeComponent.generated.h"

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

USTRUCT(BlueprintType)
struct FDroneViewpointSnapshotNative
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FString timestamp_utc;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FString run_id;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FString capture_id;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FName viewpoint_name = FName(TEXT("front"));

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    double fov_deg = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    int32 width = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    int32 height = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FVector rig_offset_from_drone_body_cm = FVector::ZeroVector;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors")
    FVector rig_rotation_from_drone_body_deg = FVector::ZeroVector;
};

UCLASS(BlueprintType, Blueprintable, ClassGroup = (Drone), meta = (BlueprintSpawnableComponent))
class UE_DRONE_ENV_API UDroneSensorsRuntimeComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UDroneSensorsRuntimeComponent();

    UFUNCTION(BlueprintCallable, Category = "Drone|Sensors")
    void ApplySensorRigConfig();

    UFUNCTION(BlueprintCallable, Category = "Drone|Sensors")
    bool CaptureActiveViewpointPngBytes(TArray<uint8>& OutPngBytes, int32& OutWidth, int32& OutHeight, FString& OutError);

    UFUNCTION(BlueprintCallable, Category = "Drone|Sensors")
    bool GetViewpointSnapshot(
        const FString& RunId,
        const FString& CaptureId,
        FDroneViewpointSnapshotNative& OutSnapshot,
        FString& OutError
    ) const;

    UFUNCTION(BlueprintCallable, Category = "Drone|Sensors")
    TArray<FName> ListViewpoints() const;

protected:
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors|Validation")
    bool bRequireFrontOnlyViewpoint = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors|Validation", meta = (ClampMin = "1.0", ClampMax = "179.0"))
    double MinAllowedFovDeg = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors|Validation", meta = (ClampMin = "1.0", ClampMax = "179.0"))
    double MaxAllowedFovDeg = 179.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Drone|Sensors|Validation", meta = (ClampMin = "1"))
    int32 MaxAllowedCaptureDimension = 8192;

    UPROPERTY(Transient)
    TObjectPtr<USceneCaptureComponent2D> RuntimeCaptureComponent;

    UPROPERTY(Transient)
    TObjectPtr<UTextureRenderTarget2D> RuntimeCaptureRenderTarget;

private:
    bool TryReadNameLikeProperty(const TCHAR* PropertyName, FName& OutValue) const;
    bool TryReadRealLikeProperty(const TCHAR* PropertyName, double& OutValue) const;
    bool TryReadIntLikeProperty(const TCHAR* PropertyName, int32& OutValue) const;
    bool SetBoolLikeProperty(const TCHAR* PropertyName, bool Value);
    bool SetStringLikeProperty(const TCHAR* PropertyName, const FString& Value);
    bool ReadActiveSensorRigState(
        FName& OutActiveViewpoint,
        double& OutFrontFovDeg,
        int32& OutCaptureWidth,
        int32& OutCaptureHeight,
        FVector& OutFrontOffsetCm,
        FRotator& OutFrontRotationDeg,
        FString& OutError
    ) const;
    bool EnsureCaptureResources(int32 CaptureWidth, int32 CaptureHeight, double FrontFovDeg, FString& OutError);
    bool ReadPngFromRenderTarget(TArray<uint8>& OutPngBytes, int32& OutWidth, int32& OutHeight, FString& OutError) const;
    void RecordApplyResult(bool bSuccess, const FString& ErrorMessage);
};
