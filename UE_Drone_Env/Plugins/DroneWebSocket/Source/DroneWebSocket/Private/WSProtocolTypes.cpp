#include "WSProtocolTypes.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
    static constexpr TCHAR SchemaVersion[] = TEXT("1.0");

    static TSharedPtr<FJsonObject> ParseJsonObject(const FString& JsonText)
    {
        TSharedPtr<FJsonObject> JsonObject;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
        if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
        {
            return nullptr;
        }
        return JsonObject;
    }
}

FString UWSProtocolTypes::BuildEnvelopeJson(
    const FString& Type,
    const FString& RunId,
    const FString& PayloadJson,
    int32 Seq,
    const FString& DroneId,
    const FString& CaptureId
)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("type"), Type);
    Root->SetStringField(TEXT("run_id"), RunId);
    Root->SetStringField(TEXT("schema_version"), SchemaVersion);
    Root->SetNumberField(TEXT("seq"), Seq);
    Root->SetStringField(TEXT("timestamp"), FDateTime::UtcNow().ToIso8601());

    if (!DroneId.IsEmpty())
    {
        Root->SetStringField(TEXT("drone_id"), DroneId);
    }
    if (!CaptureId.IsEmpty())
    {
        Root->SetStringField(TEXT("capture_id"), CaptureId);
    }
    Root->SetStringField(TEXT("message_id"), FGuid::NewGuid().ToString(EGuidFormats::DigitsWithHyphensLower));

    TSharedPtr<FJsonObject> PayloadObject = ParseJsonObject(PayloadJson);
    if (!PayloadObject.IsValid())
    {
        PayloadObject = MakeShared<FJsonObject>();
        PayloadObject->SetStringField(TEXT("raw"), PayloadJson);
    }
    Root->SetObjectField(TEXT("payload"), PayloadObject.ToSharedRef());

    FString OutJson;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutJson);
    FJsonSerializer::Serialize(Root, Writer);
    return OutJson;
}

bool UWSProtocolTypes::ParseEnvelopeJson(
    const FString& MessageJson,
    FWSBridgeEnvelope& OutEnvelope,
    FString& OutError
)
{
    const TSharedPtr<FJsonObject> Root = ParseJsonObject(MessageJson);
    if (!Root.IsValid())
    {
        OutError = TEXT("Failed to parse message JSON.");
        return false;
    }

    if (!Root->TryGetStringField(TEXT("type"), OutEnvelope.Type) || OutEnvelope.Type.IsEmpty())
    {
        OutError = TEXT("Missing required field: type");
        return false;
    }
    if (!Root->TryGetStringField(TEXT("run_id"), OutEnvelope.RunId) || OutEnvelope.RunId.IsEmpty())
    {
        OutError = TEXT("Missing required field: run_id");
        return false;
    }

    Root->TryGetStringField(TEXT("schema_version"), OutEnvelope.SchemaVersion);
    Root->TryGetStringField(TEXT("timestamp"), OutEnvelope.Timestamp);
    Root->TryGetStringField(TEXT("drone_id"), OutEnvelope.DroneId);
    Root->TryGetStringField(TEXT("capture_id"), OutEnvelope.CaptureId);
    Root->TryGetStringField(TEXT("message_id"), OutEnvelope.MessageId);

    int32 ParsedSeq = 0;
    Root->TryGetNumberField(TEXT("seq"), ParsedSeq);
    OutEnvelope.Seq = ParsedSeq;

    const TSharedPtr<FJsonObject>* PayloadObject = nullptr;
    if (Root->TryGetObjectField(TEXT("payload"), PayloadObject) && PayloadObject != nullptr && PayloadObject->IsValid())
    {
        FString PayloadJson;
        const TSharedRef<TJsonWriter<>> PayloadWriter = TJsonWriterFactory<>::Create(&PayloadJson);
        FJsonSerializer::Serialize(PayloadObject->ToSharedRef(), PayloadWriter);
        OutEnvelope.PayloadJson = PayloadJson;
    }
    else
    {
        OutEnvelope.PayloadJson = TEXT("{}");
    }

    OutError = TEXT("");
    return true;
}
