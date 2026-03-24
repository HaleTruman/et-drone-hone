// Status: in development.
// Native runtime orchestrator for BP_SampleManager.

#include "SampleManagerRuntimeActor.h"

#include "DroneSensorsRuntimeComponent.h"

#include "Components/ActorComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "Misc/Base64.h"
#include "Misc/Guid.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
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
    FString WantedCanonical = Wanted.ToLower();
    WantedCanonical.ReplaceInline(TEXT("_"), TEXT(""));
    for (TFieldIterator<FProperty> It(Class, EFieldIterationFlags::IncludeSuper); It; ++It)
    {
        const FProperty* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        const FString CandidateName = Candidate->GetName();
        FString CandidateCanonical = CandidateName.ToLower();
        CandidateCanonical.ReplaceInline(TEXT("_"), TEXT(""));
        if (CandidateName.Equals(Wanted, ESearchCase::IgnoreCase)
            || CandidateName.StartsWith(WantedPrefix, ESearchCase::IgnoreCase)
            || CandidateCanonical.Equals(WantedCanonical, ESearchCase::CaseSensitive)
            || CandidateCanonical.StartsWith(WantedCanonical, ESearchCase::CaseSensitive))
        {
            return Candidate;
        }
    }

    return nullptr;
}

bool GetStringLikeProperty(const UObject* Target, const TCHAR* PropertyName, FString& OutValue)
{
    OutValue = TEXT("");
    if (Target == nullptr || PropertyName == nullptr || !*PropertyName)
    {
        return false;
    }

    const FProperty* Property = FindPropertyByStableName(Target->GetClass(), PropertyName);
    if (Property == nullptr)
    {
        return false;
    }

    if (const FStrProperty* StrProperty = CastField<FStrProperty>(Property))
    {
        OutValue = StrProperty->GetPropertyValue_InContainer(Target);
        return !OutValue.IsEmpty();
    }
    if (const FNameProperty* NameProperty = CastField<FNameProperty>(Property))
    {
        OutValue = NameProperty->GetPropertyValue_InContainer(Target).ToString();
        return !OutValue.IsEmpty();
    }
    return false;
}

TSharedPtr<FJsonObject> MakeVectorObject(const FVector& Vector)
{
    TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
    Object->SetNumberField(TEXT("x"), Vector.X);
    Object->SetNumberField(TEXT("y"), Vector.Y);
    Object->SetNumberField(TEXT("z"), Vector.Z);
    return Object;
}
}

ASampleManagerRuntimeActor::ASampleManagerRuntimeActor()
{
    PrimaryActorTick.bCanEverTick = false;
}

bool ASampleManagerRuntimeActor::ReadPrimarySensorComponentName(APawn* Pawn, FString& OutComponentName) const
{
    OutComponentName = TEXT("");
    return GetStringLikeProperty(Pawn, TEXT("primary_sensor_component_name"), OutComponentName);
}

UActorComponent* ASampleManagerRuntimeActor::ResolveSensorComponent(APawn* Pawn) const
{
    if (Pawn == nullptr)
    {
        return nullptr;
    }

    TInlineComponentArray<UActorComponent*> Components(Pawn);
    FString PreferredComponentName;
    const bool bHasPreferredName = ReadPrimarySensorComponentName(Pawn, PreferredComponentName);

    if (bHasPreferredName)
    {
        for (UActorComponent* Component : Components)
        {
            if (Component == nullptr)
            {
                continue;
            }
            if (Component->GetName().Equals(PreferredComponentName, ESearchCase::IgnoreCase))
            {
                return Component;
            }
        }
    }

    for (UActorComponent* Component : Components)
    {
        if (Component == nullptr)
        {
            continue;
        }
        const FString ComponentClassName = Component->GetClass()->GetName();
        const FString ComponentName = Component->GetName();
        if (ComponentClassName.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase)
            || ComponentName.Contains(TEXT("DroneSensors"), ESearchCase::IgnoreCase)
            || ComponentName.Contains(TEXT("SensorsModule"), ESearchCase::IgnoreCase))
        {
            return Component;
        }
    }

    return nullptr;
}

bool ASampleManagerRuntimeActor::ResolveDronePawn(const FString& DroneId, APawn*& OutPawn, FString& OutError) const
{
    OutPawn = nullptr;
    OutError = TEXT("");

    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        OutError = TEXT("Capture failed: world is null.");
        return false;
    }

    TArray<APawn*> Pawns;
    for (TActorIterator<APawn> It(World); It; ++It)
    {
        if (APawn* Pawn = *It)
        {
            Pawns.Add(Pawn);
        }
    }

    if (Pawns.Num() == 0)
    {
        OutError = TEXT("Capture failed: no pawn actors are present.");
        return false;
    }

    if (DroneId.IsEmpty())
    {
        if (Pawns.Num() == 1)
        {
            OutPawn = Pawns[0];
            return true;
        }
        OutError = FString::Printf(
            TEXT("Capture failed: drone_id is required when %d pawns exist."),
            Pawns.Num()
        );
        return false;
    }

    const FString ExpectedTag = FString::Printf(TEXT("DroneId:%s"), *DroneId);
    for (APawn* Pawn : Pawns)
    {
        if (Pawn == nullptr)
        {
            continue;
        }
        for (const FName& Tag : Pawn->Tags)
        {
            if (Tag.ToString().Equals(ExpectedTag, ESearchCase::CaseSensitive))
            {
                OutPawn = Pawn;
                return true;
            }
        }
    }

    OutError = FString::Printf(TEXT("Capture failed: no pawn found for drone_id=%s."), *DroneId);
    return false;
}

AActor* ASampleManagerRuntimeActor::ResolveConfigReferenceActor(FString& OutError) const
{
    OutError = TEXT("");
    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        OutError = TEXT("Config reference resolve failed: world is null.");
        return nullptr;
    }

    const FString FunctionName = config_reference_read_function.IsEmpty()
        ? TEXT("GetActiveConfigReference")
        : config_reference_read_function;
    const FName FunctionFName(*FunctionName);

    TArray<AActor*> FunctionCandidates;
#if WITH_EDITOR
    TArray<AActor*> LabelCandidates;
#endif

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        if (Candidate->FindFunction(FunctionFName) == nullptr)
        {
            continue;
        }
        FunctionCandidates.Add(Candidate);

#if WITH_EDITOR
        if (!config_reference_source_actor_label.IsEmpty()
            && Candidate->GetActorLabel().Equals(config_reference_source_actor_label, ESearchCase::IgnoreCase))
        {
            LabelCandidates.Add(Candidate);
        }
#endif
    }

#if WITH_EDITOR
    if (!config_reference_source_actor_label.IsEmpty())
    {
        if (LabelCandidates.Num() == 1)
        {
            return LabelCandidates[0];
        }
        if (LabelCandidates.Num() > 1)
        {
            OutError = FString::Printf(
                TEXT("Config reference resolve failed: duplicate ingress actors for label=%s count=%d."),
                *config_reference_source_actor_label,
                LabelCandidates.Num()
            );
            return nullptr;
        }
        OutError = FString::Printf(
            TEXT("Config reference resolve failed: no ingress actor found for label=%s."),
            *config_reference_source_actor_label
        );
        return nullptr;
    }
#endif

    if (FunctionCandidates.Num() == 1)
    {
        return FunctionCandidates[0];
    }
    if (FunctionCandidates.Num() == 0)
    {
        OutError = FString::Printf(
            TEXT("Config reference resolve failed: no actor exposing %s."),
            *FunctionName
        );
        return nullptr;
    }

    OutError = FString::Printf(
        TEXT("Config reference resolve failed: multiple actors expose %s count=%d."),
        *FunctionName,
        FunctionCandidates.Num()
    );
    return nullptr;
}

bool ASampleManagerRuntimeActor::InvokeGetActiveConfigReference(
    UObject* Target,
    const FString& FunctionName,
    FString& OutConfigId,
    FString& OutConfigHash,
    FString& OutError
) const
{
    OutConfigId = TEXT("");
    OutConfigHash = TEXT("");
    OutError = TEXT("");

    if (Target == nullptr)
    {
        OutError = TEXT("Config reference invoke failed: target is null.");
        return false;
    }

    const FString ResolvedFunctionName = FunctionName.IsEmpty() ? TEXT("GetActiveConfigReference") : FunctionName;
    UFunction* Function = Target->FindFunction(FName(*ResolvedFunctionName));
    if (Function == nullptr)
    {
        OutError = FString::Printf(
            TEXT("Config reference invoke failed: function %s not found on %s."),
            *ResolvedFunctionName,
            *Target->GetName()
        );
        return false;
    }

    TArray<uint8> ParamBuffer;
    ParamBuffer.SetNumZeroed(Function->ParmsSize);
    Function->InitializeStruct(ParamBuffer.GetData());
    Target->ProcessEvent(Function, ParamBuffer.GetData());

    bool bReturnValue = false;
    bool bFoundReturnValue = false;
    FString ResolvedConfigId;
    FString ResolvedConfigHash;

    for (TFieldIterator<FProperty> It(Function); It; ++It)
    {
        const FProperty* Property = *It;
        if (Property == nullptr)
        {
            continue;
        }
        if (!Property->HasAnyPropertyFlags(CPF_Parm))
        {
            continue;
        }

        const FString PropertyName = Property->GetName();
        FString CanonicalPropertyName = PropertyName.ToLower();
        CanonicalPropertyName.ReplaceInline(TEXT("_"), TEXT(""));
        const bool bIsReturn = Property->HasAnyPropertyFlags(CPF_ReturnParm);

        if (bIsReturn)
        {
            if (const FBoolProperty* BoolProperty = CastField<FBoolProperty>(Property))
            {
                bReturnValue = BoolProperty->GetPropertyValue_InContainer(ParamBuffer.GetData());
                bFoundReturnValue = true;
            }
            continue;
        }

        if (const FStrProperty* StrProperty = CastField<FStrProperty>(Property))
        {
            const FString Value = StrProperty->GetPropertyValue_InContainer(ParamBuffer.GetData());
            if (CanonicalPropertyName.Contains(TEXT("configid"), ESearchCase::CaseSensitive))
            {
                ResolvedConfigId = Value;
            }
            else if (CanonicalPropertyName.Contains(TEXT("confighash"), ESearchCase::CaseSensitive))
            {
                ResolvedConfigHash = Value;
            }
            continue;
        }

        if (const FNameProperty* NameProperty = CastField<FNameProperty>(Property))
        {
            const FString Value = NameProperty->GetPropertyValue_InContainer(ParamBuffer.GetData()).ToString();
            if (CanonicalPropertyName.Contains(TEXT("configid"), ESearchCase::CaseSensitive))
            {
                ResolvedConfigId = Value;
            }
            else if (CanonicalPropertyName.Contains(TEXT("confighash"), ESearchCase::CaseSensitive))
            {
                ResolvedConfigHash = Value;
            }
            continue;
        }
    }

    if (ResolvedConfigId.IsEmpty() || ResolvedConfigHash.IsEmpty())
    {
        FString PropertyConfigId;
        FString PropertyConfigHash;
        const bool bHasConfigId = GetStringLikeProperty(Target, TEXT("active_config_id"), PropertyConfigId);
        const bool bHasConfigHash = GetStringLikeProperty(Target, TEXT("active_config_hash"), PropertyConfigHash);
        if (bHasConfigId && bHasConfigHash && !PropertyConfigId.IsEmpty() && !PropertyConfigHash.IsEmpty())
        {
            ResolvedConfigId = PropertyConfigId;
            ResolvedConfigHash = PropertyConfigHash;
        }
    }

    if (ResolvedConfigId.IsEmpty() || ResolvedConfigHash.IsEmpty())
    {
        Function->DestroyStruct(ParamBuffer.GetData());
        if (bFoundReturnValue && !bReturnValue)
        {
            OutError = TEXT("Config reference invoke failed: ingress reported no active config.");
        }
        else
        {
            OutError = TEXT("Config reference invoke failed: config_id/config_hash was empty.");
        }
        return false;
    }

    Function->DestroyStruct(ParamBuffer.GetData());
    OutConfigId = ResolvedConfigId;
    OutConfigHash = ResolvedConfigHash;
    return true;
}

bool ASampleManagerRuntimeActor::ResolveActiveConfigReference(
    FString& OutConfigId,
    FString& OutConfigHash,
    FString& OutError
) const
{
    OutConfigId = TEXT("");
    OutConfigHash = TEXT("");
    OutError = TEXT("");

    const FString FunctionName = config_reference_read_function.IsEmpty()
        ? TEXT("GetActiveConfigReference")
        : config_reference_read_function;
    const FName FunctionFName(*FunctionName);

    FString PreferredResolveError;
    AActor* PreferredSource = ResolveConfigReferenceActor(PreferredResolveError);
    if (PreferredSource != nullptr)
    {
        if (InvokeGetActiveConfigReference(PreferredSource, FunctionName, OutConfigId, OutConfigHash, OutError))
        {
            return true;
        }
    }

    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        OutError = TEXT("Config reference resolve failed: world is null.");
        return false;
    }

    AActor* SuccessfulSource = nullptr;
    FString SuccessfulConfigId;
    FString SuccessfulConfigHash;
    int32 SuccessfulSourceCount = 0;

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Candidate = *It;
        if (Candidate == nullptr || Candidate == PreferredSource)
        {
            continue;
        }
        if (Candidate->FindFunction(FunctionFName) == nullptr)
        {
            continue;
        }

        FString CandidateConfigId;
        FString CandidateConfigHash;
        FString CandidateError;
        if (!InvokeGetActiveConfigReference(
                Candidate,
                FunctionName,
                CandidateConfigId,
                CandidateConfigHash,
                CandidateError
            ))
        {
            continue;
        }

        ++SuccessfulSourceCount;
        if (SuccessfulSource == nullptr)
        {
            SuccessfulSource = Candidate;
            SuccessfulConfigId = CandidateConfigId;
            SuccessfulConfigHash = CandidateConfigHash;
            continue;
        }

        OutError = FString::Printf(
            TEXT("Config reference resolve failed: multiple active ingress sources expose %s."),
            *FunctionName
        );
        return false;
    }

    if (SuccessfulSourceCount == 1 && SuccessfulSource != nullptr)
    {
        OutConfigId = SuccessfulConfigId;
        OutConfigHash = SuccessfulConfigHash;
        OutError = TEXT("");
        return true;
    }

    if (!PreferredResolveError.IsEmpty())
    {
        OutError = PreferredResolveError;
        return false;
    }
    if (!OutError.IsEmpty())
    {
        return false;
    }

    OutError = FString::Printf(
        TEXT("Config reference resolve failed: no active source returned config for %s."),
        *FunctionName
    );
    return false;
}

bool ASampleManagerRuntimeActor::AssembleObservationJson(
    const FString& RunId,
    const FString& DroneId,
    const FString& CaptureId,
    const FString& ConfigId,
    const FString& ConfigHash,
    const FDroneViewpointSnapshotNative& Snapshot,
    const TArray<uint8>& PngBytes,
    int32 CapturedWidth,
    int32 CapturedHeight,
    FString& OutObservationJson,
    FString& OutError
) const
{
    OutObservationJson = TEXT("");
    OutError = TEXT("");

    if (PngBytes.IsEmpty())
    {
        OutError = TEXT("Observation assembly failed: image bytes are empty.");
        return false;
    }
    if (CapturedWidth <= 0 || CapturedHeight <= 0)
    {
        OutError = TEXT("Observation assembly failed: invalid captured image dimensions.");
        return false;
    }

    const FString ImageBase64 = FBase64::Encode(PngBytes);
    if (ImageBase64.IsEmpty())
    {
        OutError = TEXT("Observation assembly failed: base64 image encoding produced empty output.");
        return false;
    }

    TSharedPtr<FJsonObject> ConfigRefObject = MakeShared<FJsonObject>();
    ConfigRefObject->SetStringField(TEXT("config_id"), ConfigId);
    ConfigRefObject->SetStringField(TEXT("config_hash"), ConfigHash);

    TSharedPtr<FJsonObject> ViewpointObject = MakeShared<FJsonObject>();
    ViewpointObject->SetStringField(TEXT("timestamp_utc"), Snapshot.timestamp_utc);
    ViewpointObject->SetStringField(TEXT("run_id"), Snapshot.run_id);
    ViewpointObject->SetStringField(TEXT("capture_id"), Snapshot.capture_id);
    ViewpointObject->SetStringField(TEXT("viewpoint_name"), Snapshot.viewpoint_name.ToString());
    ViewpointObject->SetNumberField(TEXT("fov_deg"), Snapshot.fov_deg);
    ViewpointObject->SetNumberField(TEXT("width"), Snapshot.width);
    ViewpointObject->SetNumberField(TEXT("height"), Snapshot.height);
    ViewpointObject->SetObjectField(
        TEXT("rig_offset_from_drone_body_cm"),
        MakeVectorObject(Snapshot.rig_offset_from_drone_body_cm)
    );
    ViewpointObject->SetObjectField(
        TEXT("rig_rotation_from_drone_body_deg"),
        MakeVectorObject(Snapshot.rig_rotation_from_drone_body_deg)
    );

    TSharedPtr<FJsonObject> CanonicalImageObject = MakeShared<FJsonObject>();
    CanonicalImageObject->SetStringField(TEXT("encoding"), TEXT("png_base64"));
    CanonicalImageObject->SetStringField(TEXT("bytes_b64"), ImageBase64);
    CanonicalImageObject->SetNumberField(TEXT("width"), CapturedWidth);
    CanonicalImageObject->SetNumberField(TEXT("height"), CapturedHeight);

    TSharedPtr<FJsonObject> ObservationObject = MakeShared<FJsonObject>();
    ObservationObject->SetStringField(TEXT("timestamp_utc"), Snapshot.timestamp_utc);
    ObservationObject->SetStringField(TEXT("run_id"), RunId);
    ObservationObject->SetStringField(TEXT("drone_id"), DroneId);
    ObservationObject->SetStringField(TEXT("capture_id"), CaptureId);
    ObservationObject->SetObjectField(TEXT("config_ref"), ConfigRefObject);
    ObservationObject->SetObjectField(TEXT("viewpoint"), ViewpointObject);
    ObservationObject->SetObjectField(TEXT("image"), CanonicalImageObject);
    ObservationObject->SetStringField(TEXT("image_bytes_b64"), ImageBase64);

    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutObservationJson);
    if (!FJsonSerializer::Serialize(ObservationObject.ToSharedRef(), Writer))
    {
        OutObservationJson = TEXT("");
        OutError = TEXT("Observation assembly failed: JSON serialization failed.");
        return false;
    }

    return true;
}

bool ASampleManagerRuntimeActor::CaptureNow(
    const FString& RunId,
    const FString& DroneId,
    const FString& CaptureId,
    FString& OutObservationJson,
    FString& OutError
)
{
    OutObservationJson = TEXT("");
    OutError = TEXT("");

    if (RunId.IsEmpty())
    {
        OutError = TEXT("Capture failed: run_id is empty.");
        return false;
    }

    APawn* TargetPawn = nullptr;
    FString ResolvePawnError;
    if (!ResolveDronePawn(DroneId, TargetPawn, ResolvePawnError))
    {
        OutError = ResolvePawnError;
        return false;
    }

    UActorComponent* SensorComponent = ResolveSensorComponent(TargetPawn);
    if (SensorComponent == nullptr)
    {
        OutError = FString::Printf(
            TEXT("Capture failed: no hosted sensor component was resolved for pawn=%s."),
            *TargetPawn->GetName()
        );
        return false;
    }

    UDroneSensorsRuntimeComponent* SensorRuntime = Cast<UDroneSensorsRuntimeComponent>(SensorComponent);
    if (SensorRuntime == nullptr)
    {
        OutError = FString::Printf(
            TEXT("Capture failed: resolved sensor component is not DroneSensorsRuntimeComponent class=%s name=%s."),
            *SensorComponent->GetClass()->GetName(),
            *SensorComponent->GetName()
        );
        return false;
    }

    FString ActiveConfigId;
    FString ActiveConfigHash;
    FString ConfigRefError;
    if (!ResolveActiveConfigReference(ActiveConfigId, ActiveConfigHash, ConfigRefError))
    {
        OutError = ConfigRefError;
        return false;
    }

    const FString ResolvedCaptureId = CaptureId.IsEmpty()
        ? FString::Printf(TEXT("capture_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(12))
        : CaptureId;

    TArray<uint8> PngBytes;
    int32 CapturedWidth = 0;
    int32 CapturedHeight = 0;
    FString CaptureError;
    if (!SensorRuntime->CaptureActiveViewpointPngBytes(PngBytes, CapturedWidth, CapturedHeight, CaptureError))
    {
        OutError = FString::Printf(TEXT("Capture failed: sensor image capture failed: %s"), *CaptureError);
        return false;
    }

    FDroneViewpointSnapshotNative Snapshot;
    FString SnapshotError;
    if (!SensorRuntime->GetViewpointSnapshot(RunId, ResolvedCaptureId, Snapshot, SnapshotError))
    {
        OutError = FString::Printf(TEXT("Capture failed: viewpoint snapshot failed: %s"), *SnapshotError);
        return false;
    }

    if (Snapshot.width != CapturedWidth || Snapshot.height != CapturedHeight)
    {
        OutError = FString::Printf(
            TEXT("Capture failed: snapshot/capture dimension mismatch snapshot=%dx%d capture=%dx%d."),
            Snapshot.width,
            Snapshot.height,
            CapturedWidth,
            CapturedHeight
        );
        return false;
    }
    if (Snapshot.run_id != RunId || Snapshot.capture_id != ResolvedCaptureId)
    {
        OutError = FString::Printf(
            TEXT("Capture failed: snapshot identity mismatch run_id=%s/%s capture_id=%s/%s."),
            *Snapshot.run_id,
            *RunId,
            *Snapshot.capture_id,
            *ResolvedCaptureId
        );
        return false;
    }

    const FString ResolvedDroneId = DroneId.IsEmpty() ? TEXT("") : DroneId;
    return AssembleObservationJson(
        RunId,
        ResolvedDroneId,
        ResolvedCaptureId,
        ActiveConfigId,
        ActiveConfigHash,
        Snapshot,
        PngBytes,
        CapturedWidth,
        CapturedHeight,
        OutObservationJson,
        OutError
    );
}

bool ASampleManagerRuntimeActor::CaptureNowTransport(FString RunId, FString DroneId, FString CaptureId)
{
    transport_last_success = false;
    transport_last_observation_json = TEXT("");
    transport_last_error = TEXT("");
    transport_last_drone_id = DroneId;
    transport_last_capture_id = CaptureId;

    FString ObservationJson;
    FString CaptureError;
    const bool bSuccess = CaptureNow(RunId, DroneId, CaptureId, ObservationJson, CaptureError);

    transport_last_success = bSuccess;
    transport_last_observation_json = ObservationJson;
    transport_last_error = CaptureError;

    if (bSuccess)
    {
        TSharedPtr<FJsonObject> ObservationObject;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ObservationJson);
        if (FJsonSerializer::Deserialize(Reader, ObservationObject) && ObservationObject.IsValid())
        {
            FString ResolvedDroneId;
            if (ObservationObject->TryGetStringField(TEXT("drone_id"), ResolvedDroneId) && !ResolvedDroneId.IsEmpty())
            {
                transport_last_drone_id = ResolvedDroneId;
            }
            FString ResolvedCaptureId;
            if (ObservationObject->TryGetStringField(TEXT("capture_id"), ResolvedCaptureId) && !ResolvedCaptureId.IsEmpty())
            {
                transport_last_capture_id = ResolvedCaptureId;
            }
        }
    }

    return bSuccess;
}

bool ASampleManagerRuntimeActor::CaptureNowTransportEvent_Implementation(
    const FString& RunId,
    const FString& DroneId,
    const FString& CaptureId
)
{
    return CaptureNowTransport(RunId, DroneId, CaptureId);
}
