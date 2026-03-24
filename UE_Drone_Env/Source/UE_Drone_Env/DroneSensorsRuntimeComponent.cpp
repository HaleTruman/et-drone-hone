// Status: in development.
// Provides native runtime sensor-rig apply validation plus callable capture and
// viewpoint snapshot functions for BP_DroneSensors.

#include "DroneSensorsRuntimeComponent.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "ImageUtils.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Char.h"
#include "UObject/UnrealType.h"

namespace
{
const FProperty* FindPropertyByStableName(const UClass* Class, const TCHAR* PropertyName)
{
    if (Class == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return nullptr;
    }

    if (const FProperty* Exact = Class->FindPropertyByName(FName(PropertyName)))
    {
        return Exact;
    }

    const FString Wanted(PropertyName);
    const FString WantedPrefix = Wanted + TEXT("_");
    for (TFieldIterator<FProperty> It(Class, EFieldIterationFlags::IncludeSuper); It; ++It)
    {
        const FProperty* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        const FString CandidateName = Candidate->GetName();
        if (CandidateName.Equals(Wanted, ESearchCase::IgnoreCase)
            || CandidateName.StartsWith(WantedPrefix, ESearchCase::IgnoreCase))
        {
            return Candidate;
        }
    }

    return nullptr;
}

FProperty* FindPropertyByStableName(UClass* Class, const TCHAR* PropertyName)
{
    return const_cast<FProperty*>(FindPropertyByStableName(static_cast<const UClass*>(Class), PropertyName));
}
}

UDroneSensorsRuntimeComponent::UDroneSensorsRuntimeComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UDroneSensorsRuntimeComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (RuntimeCaptureComponent != nullptr)
    {
        RuntimeCaptureComponent->DestroyComponent();
        RuntimeCaptureComponent = nullptr;
    }
    RuntimeCaptureRenderTarget = nullptr;
    Super::EndPlay(EndPlayReason);
}

void UDroneSensorsRuntimeComponent::ApplySensorRigConfig()
{
    FName ActiveViewpoint;
    double FrontFovDeg = 0.0;
    int32 CaptureWidth = 0;
    int32 CaptureHeight = 0;
    FVector FrontOffsetCm = FVector::ZeroVector;
    FRotator FrontRotationDeg = FRotator::ZeroRotator;
    FString ValidationError;

    if (!ReadActiveSensorRigState(
            ActiveViewpoint,
            FrontFovDeg,
            CaptureWidth,
            CaptureHeight,
            FrontOffsetCm,
            FrontRotationDeg,
            ValidationError
        ))
    {
        RecordApplyResult(false, ValidationError);
        return;
    }

    RecordApplyResult(true, TEXT(""));
}

bool UDroneSensorsRuntimeComponent::CaptureActiveViewpointPngBytes(
    TArray<uint8>& OutPngBytes,
    int32& OutWidth,
    int32& OutHeight,
    FString& OutError
)
{
    OutPngBytes.Reset();
    OutWidth = 0;
    OutHeight = 0;
    OutError = TEXT("");

    FName ActiveViewpoint;
    double FrontFovDeg = 0.0;
    int32 CaptureWidth = 0;
    int32 CaptureHeight = 0;
    FVector FrontOffsetCm = FVector::ZeroVector;
    FRotator FrontRotationDeg = FRotator::ZeroRotator;
    if (!ReadActiveSensorRigState(
            ActiveViewpoint,
            FrontFovDeg,
            CaptureWidth,
            CaptureHeight,
            FrontOffsetCm,
            FrontRotationDeg,
            OutError
        ))
    {
        return false;
    }

    if (!EnsureCaptureResources(CaptureWidth, CaptureHeight, FrontFovDeg, OutError))
    {
        return false;
    }

    AActor* Owner = GetOwner();
    if (Owner == nullptr)
    {
        OutError = TEXT("Capture failed: owner actor is null.");
        return false;
    }

    const FTransform OwnerTransform = Owner->GetActorTransform();
    const FVector WorldLocation = OwnerTransform.TransformPosition(FrontOffsetCm);
    const FQuat WorldRotationQuat = OwnerTransform.GetRotation() * FrontRotationDeg.Quaternion();
    const FRotator WorldRotation = WorldRotationQuat.Rotator();

    RuntimeCaptureComponent->SetWorldLocation(WorldLocation);
    RuntimeCaptureComponent->SetWorldRotation(WorldRotation);
    RuntimeCaptureComponent->FOVAngle = static_cast<float>(FrontFovDeg);
    RuntimeCaptureComponent->CaptureScene();

    if (!ReadPngFromRenderTarget(OutPngBytes, OutWidth, OutHeight, OutError))
    {
        return false;
    }

    if (OutWidth != CaptureWidth || OutHeight != CaptureHeight)
    {
        OutError = FString::Printf(
            TEXT("Captured PNG dimensions mismatch expected=%dx%d actual=%dx%d"),
            CaptureWidth,
            CaptureHeight,
            OutWidth,
            OutHeight
        );
        return false;
    }

    return true;
}

bool UDroneSensorsRuntimeComponent::GetViewpointSnapshot(
    const FString& RunId,
    const FString& CaptureId,
    FDroneViewpointSnapshotNative& OutSnapshot,
    FString& OutError
) const
{
    OutError = TEXT("");

    FName ActiveViewpoint;
    double FrontFovDeg = 0.0;
    int32 CaptureWidth = 0;
    int32 CaptureHeight = 0;
    FVector FrontOffsetCm = FVector::ZeroVector;
    FRotator FrontRotationDeg = FRotator::ZeroRotator;
    if (!ReadActiveSensorRigState(
            ActiveViewpoint,
            FrontFovDeg,
            CaptureWidth,
            CaptureHeight,
            FrontOffsetCm,
            FrontRotationDeg,
            OutError
        ))
    {
        return false;
    }

    OutSnapshot.timestamp_utc = FDateTime::UtcNow().ToIso8601();
    OutSnapshot.run_id = RunId;
    OutSnapshot.capture_id = CaptureId;
    OutSnapshot.viewpoint_name = ActiveViewpoint;
    OutSnapshot.fov_deg = FrontFovDeg;
    OutSnapshot.width = CaptureWidth;
    OutSnapshot.height = CaptureHeight;
    OutSnapshot.rig_offset_from_drone_body_cm = FrontOffsetCm;
    OutSnapshot.rig_rotation_from_drone_body_deg = FVector(
        FrontRotationDeg.Pitch,
        FrontRotationDeg.Roll,
        FrontRotationDeg.Yaw
    );
    return true;
}

TArray<FName> UDroneSensorsRuntimeComponent::ListViewpoints() const
{
    FName ActiveViewpoint;
    if (TryReadNameLikeProperty(TEXT("active_viewpoint"), ActiveViewpoint) && !ActiveViewpoint.IsNone())
    {
        TArray<FName> Single;
        Single.Add(ActiveViewpoint);
        return Single;
    }

    TArray<FName> FrontOnly;
    FrontOnly.Add(FName(TEXT("front")));
    return FrontOnly;
}

bool UDroneSensorsRuntimeComponent::TryReadNameLikeProperty(const TCHAR* PropertyName, FName& OutValue) const
{
    if (PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    const FProperty* Property = FindPropertyByStableName(GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (const FNameProperty* NameProperty = CastField<FNameProperty>(Property))
    {
        OutValue = NameProperty->GetPropertyValue_InContainer(this);
        return true;
    }
    if (const FStrProperty* StringProperty = CastField<FStrProperty>(Property))
    {
        OutValue = FName(*StringProperty->GetPropertyValue_InContainer(this));
        return true;
    }
    return false;
}

bool UDroneSensorsRuntimeComponent::TryReadRealLikeProperty(const TCHAR* PropertyName, double& OutValue) const
{
    if (PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    const FProperty* Property = FindPropertyByStableName(GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (const FFloatProperty* FloatProperty = CastField<FFloatProperty>(Property))
    {
        OutValue = static_cast<double>(FloatProperty->GetPropertyValue_InContainer(this));
        return true;
    }
    if (const FDoubleProperty* DoubleProperty = CastField<FDoubleProperty>(Property))
    {
        OutValue = DoubleProperty->GetPropertyValue_InContainer(this);
        return true;
    }
    return false;
}

bool UDroneSensorsRuntimeComponent::TryReadIntLikeProperty(const TCHAR* PropertyName, int32& OutValue) const
{
    if (PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    const FProperty* Property = FindPropertyByStableName(GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (const FIntProperty* IntProperty = CastField<FIntProperty>(Property))
    {
        OutValue = IntProperty->GetPropertyValue_InContainer(this);
        return true;
    }
    if (const FInt64Property* Int64Property = CastField<FInt64Property>(Property))
    {
        OutValue = static_cast<int32>(Int64Property->GetPropertyValue_InContainer(this));
        return true;
    }
    return false;
}

bool UDroneSensorsRuntimeComponent::SetBoolLikeProperty(const TCHAR* PropertyName, bool Value)
{
    if (PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    FProperty* Property = FindPropertyByStableName(GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (FBoolProperty* BoolProperty = CastField<FBoolProperty>(Property))
    {
        BoolProperty->SetPropertyValue_InContainer(this, Value);
        return true;
    }
    return false;
}

bool UDroneSensorsRuntimeComponent::SetStringLikeProperty(const TCHAR* PropertyName, const FString& Value)
{
    if (PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }
    FProperty* Property = FindPropertyByStableName(GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }
    if (FStrProperty* StringProperty = CastField<FStrProperty>(Property))
    {
        StringProperty->SetPropertyValue_InContainer(this, Value);
        return true;
    }
    return false;
}

bool UDroneSensorsRuntimeComponent::ReadActiveSensorRigState(
    FName& OutActiveViewpoint,
    double& OutFrontFovDeg,
    int32& OutCaptureWidth,
    int32& OutCaptureHeight,
    FVector& OutFrontOffsetCm,
    FRotator& OutFrontRotationDeg,
    FString& OutError
) const
{
    OutError = TEXT("");

    double OffsetX = 0.0;
    double OffsetY = 0.0;
    double OffsetZ = 0.0;
    double RotationPitch = 0.0;
    double RotationRoll = 0.0;
    double RotationYaw = 0.0;

    if (!TryReadNameLikeProperty(TEXT("active_viewpoint"), OutActiveViewpoint))
    {
        OutError = TEXT("missing active_viewpoint");
        return false;
    }
    if (!TryReadRealLikeProperty(TEXT("front_fov_deg"), OutFrontFovDeg))
    {
        OutError = TEXT("missing front_fov_deg");
        return false;
    }
    if (!TryReadIntLikeProperty(TEXT("capture_width"), OutCaptureWidth))
    {
        OutError = TEXT("missing capture_width");
        return false;
    }
    if (!TryReadIntLikeProperty(TEXT("capture_height"), OutCaptureHeight))
    {
        OutError = TEXT("missing capture_height");
        return false;
    }
    if (!TryReadRealLikeProperty(TEXT("front_offset_x_cm"), OffsetX)
        || !TryReadRealLikeProperty(TEXT("front_offset_y_cm"), OffsetY)
        || !TryReadRealLikeProperty(TEXT("front_offset_z_cm"), OffsetZ))
    {
        OutError = TEXT("missing one or more front_offset_* fields");
        return false;
    }
    if (!TryReadRealLikeProperty(TEXT("front_rotation_pitch_deg"), RotationPitch)
        || !TryReadRealLikeProperty(TEXT("front_rotation_roll_deg"), RotationRoll)
        || !TryReadRealLikeProperty(TEXT("front_rotation_yaw_deg"), RotationYaw))
    {
        OutError = TEXT("missing one or more front_rotation_* fields");
        return false;
    }

    if (bRequireFrontOnlyViewpoint && !OutActiveViewpoint.ToString().Equals(TEXT("front"), ESearchCase::IgnoreCase))
    {
        OutError = FString::Printf(TEXT("unsupported active_viewpoint '%s'"), *OutActiveViewpoint.ToString());
        return false;
    }
    if (!FMath::IsFinite(OutFrontFovDeg)
        || OutFrontFovDeg < MinAllowedFovDeg
        || OutFrontFovDeg > MaxAllowedFovDeg)
    {
        OutError = FString::Printf(
            TEXT("front_fov_deg out of range value=%.6f allowed=[%.6f, %.6f]"),
            OutFrontFovDeg,
            MinAllowedFovDeg,
            MaxAllowedFovDeg
        );
        return false;
    }
    if (OutCaptureWidth <= 0
        || OutCaptureHeight <= 0
        || OutCaptureWidth > MaxAllowedCaptureDimension
        || OutCaptureHeight > MaxAllowedCaptureDimension)
    {
        OutError = FString::Printf(
            TEXT("capture dimensions out of range width=%d height=%d max=%d"),
            OutCaptureWidth,
            OutCaptureHeight,
            MaxAllowedCaptureDimension
        );
        return false;
    }
    if (!FMath::IsFinite(OffsetX)
        || !FMath::IsFinite(OffsetY)
        || !FMath::IsFinite(OffsetZ)
        || !FMath::IsFinite(RotationPitch)
        || !FMath::IsFinite(RotationRoll)
        || !FMath::IsFinite(RotationYaw))
    {
        OutError = TEXT("offset/rotation contains non-finite values");
        return false;
    }

    OutFrontOffsetCm = FVector(OffsetX, OffsetY, OffsetZ);
    OutFrontRotationDeg = FRotator(RotationPitch, RotationYaw, RotationRoll);
    return true;
}

bool UDroneSensorsRuntimeComponent::EnsureCaptureResources(
    int32 CaptureWidth,
    int32 CaptureHeight,
    double FrontFovDeg,
    FString& OutError
)
{
    OutError = TEXT("");

    AActor* Owner = GetOwner();
    if (Owner == nullptr)
    {
        OutError = TEXT("Capture failed: owner actor is null.");
        return false;
    }
    USceneComponent* RootComponent = Owner->GetRootComponent();
    if (RootComponent == nullptr)
    {
        OutError = TEXT("Capture failed: owner root component is null.");
        return false;
    }

    if (RuntimeCaptureRenderTarget == nullptr)
    {
        RuntimeCaptureRenderTarget = NewObject<UTextureRenderTarget2D>(
            this,
            TEXT("DroneSensorsRuntimeCaptureRenderTarget"),
            RF_Transient
        );
        if (RuntimeCaptureRenderTarget == nullptr)
        {
            OutError = TEXT("Failed to create runtime capture render target.");
            return false;
        }
        RuntimeCaptureRenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
        RuntimeCaptureRenderTarget->bAutoGenerateMips = false;
        RuntimeCaptureRenderTarget->ClearColor = FLinearColor::Black;
    }

    if (RuntimeCaptureRenderTarget->SizeX != CaptureWidth || RuntimeCaptureRenderTarget->SizeY != CaptureHeight)
    {
        RuntimeCaptureRenderTarget->InitAutoFormat(CaptureWidth, CaptureHeight);
        RuntimeCaptureRenderTarget->UpdateResourceImmediate(true);
    }

    if (RuntimeCaptureComponent == nullptr)
    {
        RuntimeCaptureComponent = NewObject<USceneCaptureComponent2D>(
            Owner,
            TEXT("DroneSensorsRuntimeCaptureComponent"),
            RF_Transient
        );
        if (RuntimeCaptureComponent == nullptr)
        {
            OutError = TEXT("Failed to create runtime scene capture component.");
            return false;
        }
        RuntimeCaptureComponent->SetupAttachment(RootComponent);
        RuntimeCaptureComponent->RegisterComponent();
    }

    RuntimeCaptureComponent->bCaptureEveryFrame = false;
    RuntimeCaptureComponent->bCaptureOnMovement = false;
    RuntimeCaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    RuntimeCaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;
    RuntimeCaptureComponent->FOVAngle = static_cast<float>(FrontFovDeg);
    RuntimeCaptureComponent->TextureTarget = RuntimeCaptureRenderTarget;
    return true;
}

bool UDroneSensorsRuntimeComponent::ReadPngFromRenderTarget(
    TArray<uint8>& OutPngBytes,
    int32& OutWidth,
    int32& OutHeight,
    FString& OutError
) const
{
    OutPngBytes.Reset();
    OutWidth = 0;
    OutHeight = 0;
    OutError = TEXT("");

    if (RuntimeCaptureRenderTarget == nullptr)
    {
        OutError = TEXT("Runtime capture render target is null.");
        return false;
    }

    FTextureRenderTargetResource* RenderResource = RuntimeCaptureRenderTarget->GameThread_GetRenderTargetResource();
    if (RenderResource == nullptr)
    {
        OutError = TEXT("Runtime capture render target resource is null.");
        return false;
    }

    const int32 CaptureWidth = RuntimeCaptureRenderTarget->SizeX;
    const int32 CaptureHeight = RuntimeCaptureRenderTarget->SizeY;
    if (CaptureWidth <= 0 || CaptureHeight <= 0)
    {
        OutError = TEXT("Runtime capture render target has invalid dimensions.");
        return false;
    }

    TArray<FColor> SurfaceData;
    if (!RenderResource->ReadPixels(SurfaceData))
    {
        OutError = TEXT("Failed to read pixels from runtime capture render target.");
        return false;
    }
    if (SurfaceData.Num() != CaptureWidth * CaptureHeight)
    {
        OutError = FString::Printf(
            TEXT("Unexpected surface pixel count expected=%d actual=%d"),
            CaptureWidth * CaptureHeight,
            SurfaceData.Num()
        );
        return false;
    }

    TArray64<uint8> PngBytes64;
    const TArrayView64<const FColor> SurfaceView(SurfaceData.GetData(), SurfaceData.Num());
    FImageUtils::PNGCompressImageArray(CaptureWidth, CaptureHeight, SurfaceView, PngBytes64);
    if (PngBytes64.IsEmpty())
    {
        OutError = TEXT("PNG compression produced empty image bytes.");
        return false;
    }
    if (PngBytes64.Num() > MAX_int32)
    {
        OutError = FString::Printf(
            TEXT("PNG output is too large to fit in TArray<uint8>: bytes=%lld"),
            static_cast<long long>(PngBytes64.Num())
        );
        return false;
    }

    OutPngBytes.Reset();
    OutPngBytes.Append(PngBytes64.GetData(), static_cast<int32>(PngBytes64.Num()));

    OutWidth = CaptureWidth;
    OutHeight = CaptureHeight;
    return true;
}

void UDroneSensorsRuntimeComponent::RecordApplyResult(bool bSuccess, const FString& ErrorMessage)
{
    SetBoolLikeProperty(TEXT("runtime_config_applied"), bSuccess);
    SetStringLikeProperty(TEXT("config_apply_last_error"), bSuccess ? TEXT("") : ErrorMessage);

    if (bSuccess)
    {
        UE_LOG(LogTemp, Display, TEXT("[DroneSensorsRuntimeComponent] ApplySensorRigConfig succeeded"));
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("[DroneSensorsRuntimeComponent] ApplySensorRigConfig failed: %s"), *ErrorMessage);
}
