// Desired behavior:
// - Implement minimal runtime config handshake over WebSocket.
// - Accept `SET_CONFIG`, then emit `ACK` and `CONFIG_READY` with received config identifiers.
// - Keep implementation deterministic and transport-focused for Blueprint orchestration layers.

#include "WSConfigHandshakeActor.h"

#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Pawn.h"
#include "Misc/Base64.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "WSClientComponent.h"

namespace
{
    static const FString MinimalPngBase64 = TEXT(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    );

    static double ReadNumber(const TSharedPtr<FJsonObject>& PayloadObject, const TCHAR* FieldName, double DefaultValue = 0.0)
    {
        if (!PayloadObject.IsValid())
        {
            return DefaultValue;
        }
        double Value = DefaultValue;
        if (PayloadObject->TryGetNumberField(FieldName, Value))
        {
            return Value;
        }
        return DefaultValue;
    }
}

AWSConfigHandshakeActor::AWSConfigHandshakeActor()
{
    PrimaryActorTick.bCanEverTick = false;
    ServerUrl = TEXT("ws://127.0.0.1:8765");
    WSClient = CreateDefaultSubobject<UWSClientComponent>(TEXT("WSClient"));
}

void AWSConfigHandshakeActor::BeginPlay()
{
    Super::BeginPlay();

    if (!WSClient)
    {
        LastError = TEXT("WSClient component missing.");
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *LastError);
        return;
    }

    WSClient->OnConnected.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketConnected);
    WSClient->OnClosed.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketClosed);
    WSClient->OnError.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketError);
    WSClient->OnMessage.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketMessage);

    if (bAutoConnectOnBeginPlay)
    {
        Connect();
    }
}

void AWSConfigHandshakeActor::Connect()
{
    if (!WSClient)
    {
        return;
    }
    WSClient->SetServerUrl(ServerUrl);
    WSClient->Connect();
}

void AWSConfigHandshakeActor::Disconnect()
{
    if (WSClient)
    {
        WSClient->Disconnect();
    }
}

void AWSConfigHandshakeActor::HandleSocketConnected()
{
    UE_LOG(LogTemp, Display, TEXT("[WSConfigHandshakeActor] connected url=%s"), *ServerUrl);
}

void AWSConfigHandshakeActor::HandleSocketClosed(int32 StatusCode, const FString& Reason)
{
    UE_LOG(LogTemp, Warning, TEXT("[WSConfigHandshakeActor] closed status=%d reason=%s"), StatusCode, *Reason);
}

void AWSConfigHandshakeActor::HandleSocketError(const FString& ErrorText)
{
    LastError = ErrorText;
    UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] socket error: %s"), *ErrorText);
}

void AWSConfigHandshakeActor::HandleSocketMessage(const FString& MessageText)
{
    TSharedPtr<FJsonObject> Root;
    FString ParseError;
    if (!ParseRootObject(MessageText, Root, ParseError))
    {
        LastError = ParseError;
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] parse error: %s"), *ParseError);
        return;
    }

    FString MessageType;
    FString RunId;
    if (!TryGetStringField(Root, TEXT("type"), MessageType) || !TryGetStringField(Root, TEXT("run_id"), RunId))
    {
        LastError = TEXT("Incoming message missing required fields: type/run_id.");
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *LastError);
        return;
    }

    if (!MessageType.Equals(TEXT("SET_CONFIG"), ESearchCase::IgnoreCase))
    {
        const TSharedPtr<FJsonObject>* PayloadObject = nullptr;
        if (!Root->TryGetObjectField(TEXT("payload"), PayloadObject) || PayloadObject == nullptr || !PayloadObject->IsValid())
        {
            if (MessageType.Equals(TEXT("SPAWN_DRONES"), ESearchCase::IgnoreCase)
                || MessageType.Equals(TEXT("CMD"), ESearchCase::IgnoreCase)
                || MessageType.Equals(TEXT("CAPTURE_NOW"), ESearchCase::IgnoreCase))
            {
                SendError(RunId, TEXT("Action payload object is required."));
            }
            return;
        }
        HandleActionMessage(MessageType, RunId, Root, *PayloadObject);
        return;
    }

    if (!ExpectedRunId.IsEmpty() && !RunId.Equals(ExpectedRunId, ESearchCase::CaseSensitive))
    {
        const FString ErrorText = FString::Printf(
            TEXT("RunId mismatch expected=%s received=%s"),
            *ExpectedRunId,
            *RunId
        );
        LastError = ErrorText;
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *ErrorText);
        SendError(RunId, ErrorText);
        if (bFailOnRunIdMismatch)
        {
            return;
        }
    }

    const TSharedPtr<FJsonObject>* PayloadObject = nullptr;
    if (!Root->TryGetObjectField(TEXT("payload"), PayloadObject) || PayloadObject == nullptr || !PayloadObject->IsValid())
    {
        LastError = TEXT("SET_CONFIG missing payload object.");
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *LastError);
        SendError(RunId, LastError);
        return;
    }

    FString ConfigId;
    FString ConfigHash;
    if (!TryGetStringField(*PayloadObject, TEXT("config_id"), ConfigId) || !TryGetStringField(*PayloadObject, TEXT("config_hash"), ConfigHash))
    {
        LastError = TEXT("SET_CONFIG payload missing config_id/config_hash.");
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *LastError);
        SendError(RunId, LastError);
        return;
    }

    bConfigReceived = true;
    ActiveRunId = RunId;
    ActiveConfigId = ConfigId;
    ActiveConfigHash = ConfigHash;

    SendAckAndConfigReady(RunId, ConfigId, ConfigHash);

    bConfigReady = true;
    UE_LOG(
        LogTemp,
        Display,
        TEXT("[WSConfigHandshakeActor] config ready run_id=%s config_id=%s config_hash=%s"),
        *ActiveRunId,
        *ActiveConfigId,
        *ActiveConfigHash
    );
}

void AWSConfigHandshakeActor::SendEnvelope(
    const FString& Type,
    const FString& RunId,
    const TSharedPtr<FJsonObject>& PayloadObject,
    const FString& DroneId,
    const FString& CaptureId
)
{
    if (!WSClient || !WSClient->IsConnected())
    {
        UE_LOG(LogTemp, Warning, TEXT("[WSConfigHandshakeActor] cannot send %s: websocket not connected"), *Type);
        return;
    }

    FString PayloadJson = TEXT("{}");
    if (PayloadObject.IsValid())
    {
        const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadJson);
        FJsonSerializer::Serialize(PayloadObject.ToSharedRef(), Writer);
    }

    WSClient->SendEnvelopeJson(
        Type,
        RunId,
        PayloadJson,
        OutSeq++,
        DroneId,
        CaptureId
    );
}

void AWSConfigHandshakeActor::SendAckAndConfigReady(const FString& RunId, const FString& ConfigId, const FString& ConfigHash)
{
    TSharedPtr<FJsonObject> AckPayload = MakeShared<FJsonObject>();
    AckPayload->SetBoolField(TEXT("received"), true);
    AckPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("ACK"), RunId, AckPayload);

    TSharedPtr<FJsonObject> ReadyPayload = MakeShared<FJsonObject>();
    ReadyPayload->SetStringField(TEXT("config_id"), ConfigId);
    ReadyPayload->SetStringField(TEXT("config_hash"), ConfigHash);
    ReadyPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("CONFIG_READY"), RunId, ReadyPayload);
}

void AWSConfigHandshakeActor::SendError(const FString& RunId, const FString& ErrorMessage)
{
    TSharedPtr<FJsonObject> ErrorPayload = MakeShared<FJsonObject>();
    ErrorPayload->SetStringField(TEXT("message"), ErrorMessage);
    ErrorPayload->SetStringField(TEXT("source"), TEXT("BP_SetDataConfig"));
    SendEnvelope(TEXT("ERROR"), RunId, ErrorPayload);
}

bool AWSConfigHandshakeActor::ParseRootObject(
    const FString& MessageText,
    TSharedPtr<FJsonObject>& OutRoot,
    FString& OutError
) const
{
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(MessageText);
    if (!FJsonSerializer::Deserialize(Reader, OutRoot) || !OutRoot.IsValid())
    {
        OutError = TEXT("Failed to parse message JSON.");
        return false;
    }
    OutError = TEXT("");
    return true;
}

bool AWSConfigHandshakeActor::TryGetStringField(const TSharedPtr<FJsonObject>& Root, const TCHAR* Key, FString& OutValue) const
{
    if (!Root.IsValid())
    {
        return false;
    }
    FString Parsed;
    if (!Root->TryGetStringField(Key, Parsed) || Parsed.IsEmpty())
    {
        return false;
    }
    OutValue = Parsed;
    return true;
}

void AWSConfigHandshakeActor::HandleActionMessage(
    const FString& MessageType,
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    if (!bConfigReady)
    {
        SendError(RunId, FString::Printf(TEXT("blocked pre-config-ready action: %s"), *MessageType));
        return;
    }

    if (MessageType.Equals(TEXT("SPAWN_DRONES"), ESearchCase::IgnoreCase))
    {
        HandleSpawnDrones(RunId, PayloadObject);
        return;
    }
    if (MessageType.Equals(TEXT("CMD"), ESearchCase::IgnoreCase))
    {
        HandleCmd(RunId, Root, PayloadObject);
        return;
    }
    if (MessageType.Equals(TEXT("CAPTURE_NOW"), ESearchCase::IgnoreCase))
    {
        HandleCaptureNow(RunId, Root, PayloadObject);
        return;
    }
}

void AWSConfigHandshakeActor::HandleSpawnDrones(const FString& RunId, const TSharedPtr<FJsonObject>& PayloadObject)
{
    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        SendError(RunId, TEXT("World is null during SPAWN_DRONES."));
        return;
    }

    UClass* PawnClass = StaticLoadClass(APawn::StaticClass(), nullptr, *DronePawnClassPath);
    if (PawnClass == nullptr)
    {
        SendError(RunId, FString::Printf(TEXT("Failed to load drone pawn class: %s"), *DronePawnClassPath));
        return;
    }

    int32 SpawnCount = 1;
    double ParsedCount = 1.0;
    if (PayloadObject->TryGetNumberField(TEXT("spawn_count"), ParsedCount) || PayloadObject->TryGetNumberField(TEXT("count"), ParsedCount))
    {
        SpawnCount = FMath::Max(1, static_cast<int32>(ParsedCount));
    }

    TArray<FString> RequestedIds;
    const TArray<TSharedPtr<FJsonValue>>* IdArray = nullptr;
    if (PayloadObject->TryGetArrayField(TEXT("drone_ids"), IdArray) && IdArray != nullptr)
    {
        for (const TSharedPtr<FJsonValue>& Value : *IdArray)
        {
            FString IdText;
            if (Value.IsValid() && Value->TryGetString(IdText) && !IdText.IsEmpty())
            {
                RequestedIds.Add(IdText);
            }
        }
    }

    TArray<TSharedPtr<FJsonValue>> SpawnedIdValues;
    for (int32 Index = 0; Index < SpawnCount; ++Index)
    {
        const FString DroneId = (RequestedIds.IsValidIndex(Index) && !RequestedIds[Index].IsEmpty())
            ? RequestedIds[Index]
            : NextDroneId();
        const FVector SpawnLocation(static_cast<double>(Index) * SpawnSpacingCm, 0.0, 120.0);
        APawn* SpawnedPawn = World->SpawnActor<APawn>(PawnClass, SpawnLocation, FRotator::ZeroRotator);
        if (SpawnedPawn == nullptr)
        {
            continue;
        }
        SpawnedPawn->Tags.AddUnique(FName(*FString::Printf(TEXT("DroneId:%s"), *DroneId)));
        SpawnedDrones.Add(DroneId, SpawnedPawn);
        SpawnedIdValues.Add(MakeShared<FJsonValueString>(DroneId));
    }

    TSharedPtr<FJsonObject> StatusPayload = MakeShared<FJsonObject>();
    StatusPayload->SetStringField(TEXT("event"), TEXT("SPAWN_DRONES_ACCEPTED"));
    StatusPayload->SetNumberField(TEXT("spawned_count"), SpawnedIdValues.Num());
    StatusPayload->SetArrayField(TEXT("drone_ids"), SpawnedIdValues);
    SendEnvelope(TEXT("STATUS"), RunId, StatusPayload);
}

void AWSConfigHandshakeActor::HandleCmd(
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    FString DroneId;
    if (!TryGetStringField(Root, TEXT("drone_id"), DroneId))
    {
        if (!TryGetStringField(PayloadObject, TEXT("drone_id"), DroneId))
        {
            SendError(RunId, TEXT("CMD missing drone_id."));
            return;
        }
    }

    APawn* Pawn = ResolveDroneById(DroneId);
    if (Pawn == nullptr)
    {
        SendError(RunId, FString::Printf(TEXT("CMD target not found: %s"), *DroneId));
        return;
    }

    const double Pitch = ReadNumber(PayloadObject, TEXT("pitch"), 0.0);
    const double Roll = ReadNumber(PayloadObject, TEXT("roll"), 0.0);
    const double Yaw = ReadNumber(PayloadObject, TEXT("yaw"), 0.0);
    const double Throttle = ReadNumber(PayloadObject, TEXT("throttle"), 0.0);

    const FVector TranslationDelta(
        Roll * CommandTranslationScaleCm,
        Pitch * CommandTranslationScaleCm,
        Throttle * CommandTranslationScaleCm
    );
    const FRotator RotationDelta(
        Pitch * CommandRotationScaleDeg,
        Yaw * CommandRotationScaleDeg,
        Roll * CommandRotationScaleDeg
    );

    Pawn->AddActorWorldOffset(TranslationDelta, false);
    Pawn->AddActorWorldRotation(RotationDelta);

    TSharedPtr<FJsonObject> StatusPayload = MakeShared<FJsonObject>();
    StatusPayload->SetStringField(TEXT("event"), TEXT("CMD_APPLIED"));
    StatusPayload->SetStringField(TEXT("drone_id"), DroneId);
    StatusPayload->SetNumberField(TEXT("pitch"), Pitch);
    StatusPayload->SetNumberField(TEXT("roll"), Roll);
    StatusPayload->SetNumberField(TEXT("yaw"), Yaw);
    StatusPayload->SetNumberField(TEXT("throttle"), Throttle);
    SendEnvelope(TEXT("STATUS"), RunId, StatusPayload, DroneId);
}

void AWSConfigHandshakeActor::HandleCaptureNow(
    const FString& RunId,
    const TSharedPtr<FJsonObject>& Root,
    const TSharedPtr<FJsonObject>& PayloadObject
)
{
    FString DroneId;
    if (!TryGetStringField(Root, TEXT("drone_id"), DroneId))
    {
        TryGetStringField(PayloadObject, TEXT("drone_id"), DroneId);
    }

    FString CaptureId;
    if (!TryGetStringField(Root, TEXT("capture_id"), CaptureId))
    {
        if (!TryGetStringField(PayloadObject, TEXT("capture_id"), CaptureId))
        {
            CaptureId = FString::Printf(TEXT("capture_%06d"), OutSeq);
        }
    }

    APawn* Pawn = ResolveDroneById(DroneId);
    const FVector Location = Pawn ? Pawn->GetActorLocation() : FVector::ZeroVector;
    const FRotator Rotation = Pawn ? Pawn->GetActorRotation() : FRotator::ZeroRotator;

    TSharedPtr<FJsonObject> LocationObject = MakeShared<FJsonObject>();
    LocationObject->SetNumberField(TEXT("x"), Location.X);
    LocationObject->SetNumberField(TEXT("y"), Location.Y);
    LocationObject->SetNumberField(TEXT("z"), Location.Z);

    TSharedPtr<FJsonObject> RotationObject = MakeShared<FJsonObject>();
    RotationObject->SetNumberField(TEXT("pitch"), Rotation.Pitch);
    RotationObject->SetNumberField(TEXT("yaw"), Rotation.Yaw);
    RotationObject->SetNumberField(TEXT("roll"), Rotation.Roll);

    TSharedPtr<FJsonObject> PoseObject = MakeShared<FJsonObject>();
    PoseObject->SetObjectField(TEXT("location_cm"), LocationObject);
    PoseObject->SetObjectField(TEXT("rotation_deg"), RotationObject);

    TSharedPtr<FJsonObject> ObsPayload = MakeShared<FJsonObject>();
    ObsPayload->SetStringField(TEXT("timestamp_utc"), FDateTime::UtcNow().ToIso8601());
    ObsPayload->SetStringField(TEXT("run_id"), RunId);
    ObsPayload->SetStringField(TEXT("drone_id"), DroneId);
    ObsPayload->SetStringField(TEXT("capture_id"), CaptureId);
    ObsPayload->SetObjectField(TEXT("pose"), PoseObject);
    ObsPayload->SetStringField(TEXT("image_bytes_b64"), MinimalPngBase64);
    SendEnvelope(TEXT("OBS"), RunId, ObsPayload, DroneId, CaptureId);
}

APawn* AWSConfigHandshakeActor::ResolveDroneById(const FString& DroneId) const
{
    if (!DroneId.IsEmpty())
    {
        if (const TWeakObjectPtr<APawn>* Found = SpawnedDrones.Find(DroneId))
        {
            if (Found->IsValid())
            {
                return Found->Get();
            }
        }
    }

    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        return nullptr;
    }

    for (TActorIterator<APawn> It(World); It; ++It)
    {
        APawn* Candidate = *It;
        if (Candidate == nullptr)
        {
            continue;
        }
        if (DroneId.IsEmpty())
        {
            return Candidate;
        }
        const FString ExpectedTag = FString::Printf(TEXT("DroneId:%s"), *DroneId);
        for (const FName& Tag : Candidate->Tags)
        {
            if (Tag.ToString().Equals(ExpectedTag, ESearchCase::CaseSensitive))
            {
                return Candidate;
            }
        }
    }
    return nullptr;
}

FString AWSConfigHandshakeActor::NextDroneId() const
{
    int32 Index = 1;
    while (true)
    {
        const FString Candidate = FString::Printf(TEXT("drone_%04d"), Index);
        if (!SpawnedDrones.Contains(Candidate))
        {
            return Candidate;
        }
        ++Index;
    }
}
